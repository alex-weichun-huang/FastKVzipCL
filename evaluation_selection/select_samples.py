"""Replay the completed runs' evaluation IDs using pinned public datasets.

Standard-library-only until materialization; downloads data/tokenizer, never LLM weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = ('squad', 'mrcr', 'scbench_summary')
OUTPUTS = tuple(f'{name}.jsonl' for name in TASKS) + ('scbench_split_ids.json',)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_spec(config_path, ids_path):
    config, lock = read(config_path), read(ids_path)
    require(file_digest(ids_path) == config['lock_sha256'], 'Selection lock checksum changed')
    groups = []
    for split in ('train', 'validation', 'test'):
        rows = lock['scbench_splits'][split]
        require(len(rows) == config['scbench_split'][split], f'Wrong {split} count')
        groups.append({row['context_hash'] for row in rows})
        require(len(groups[-1]) == len(rows), 'Duplicate context within SCBench split')
    require(all(not groups[i] & groups[j] for i in range(3) for j in range(i)), 'SCBench split overlap')
    for task in TASKS:
        rows, spec = lock['tasks'][task], config['tasks'][task]
        require(len(rows) == spec['contexts'], f'Wrong {task} count')
        require(sum(len(row['question_ids']) for row in rows) == spec['questions'], f'Wrong {task} question count')
        require(len({r['context_hash'] for r in rows}) == len(rows), f'Duplicate {task} context')
        require(not {r['context_hash'] for r in rows} & (groups[0] | groups[1]), 'Training/validation context in final selection')
    require({r['context_hash'] for r in lock['tasks']['scbench_summary']} == groups[2], 'SCBench evaluation differs from locked test split')
    return config, lock


def verify_rows(rows, locked):
    require(len(rows) == len(locked), 'Context count changed')
    for row, expected in zip(rows, locked):
        require(digest(row['context']) == expected['context_hash'], 'Context content changed')
        for field in ('id', 'source_row', 'source_id', 'context_hash', 'context_tokens', 'n_needles'):
            if field in expected:
                require(row.get(field) == expected[field], f'Changed source field: {field}')
        require([turn['id'] for turn in row['turns']] == expected['question_ids'], 'Question IDs/order changed')


def verify_existing(directory, config, lock):
    """Check final run manifests without downloads/tokenization; read-only."""
    for task in TASKS:
        rows = [json.loads(line) for line in (directory / f'full_{task}.jsonl').read_text().splitlines()]
        # Original SCBench evaluation manifests omit token counts; the split records
        # carry them, and they are checked separately below.
        expected = [{k: v for k, v in r.items() if task != 'scbench_summary' or k != 'context_tokens'} for r in lock['tasks'][task]]
        verify_rows(rows, expected)
        require(all(r['revision'] == config['tasks'][task]['revision'] for r in rows), 'Dataset revision changed')
    original = read(directory / 'protocol.json')['splits']
    for name, rows in lock['scbench_splits'].items():
        require(len(original[name]) == len(rows), 'Split count changed')
        require([{k: r[k] for k in expected} for r, expected in zip(original[name], rows)] == rows, 'Split IDs/order changed')


def flatten_mrcr(sample):
    messages = json.loads(sample['prompt'])
    context = ''.join(('User: ' if m['role'] == 'user' else 'Assistant: ') + m['content'] + '\n\n' for m in messages[:-1])
    return context, messages[-1]['content']


def materialize(config, lock, tokenizer, loader):
    """Load only data and tokenizer; loader injection keeps tests lightweight."""
    spec = config['tasks']['squad']
    dataset = loader(spec['dataset'], spec['config'], split=spec['split'], revision=spec['revision'])
    groups = {row['context_hash']: [] for row in lock['tasks']['squad']}
    first_rows = {}
    for source_row, raw in enumerate(dataset):
        key = digest(raw['context'])
        if key in groups:
            first_rows.setdefault(key, source_row)
            groups[key].append(raw)
    result = {'squad': []}
    for expected in lock['tasks']['squad']:
        raw = groups[expected['context_hash']]
        require(bool(raw), 'Missing SQuAD context')
        result['squad'].append(dict(
            id=expected['id'], source_row=first_rows[expected['context_hash']], context_hash=expected['context_hash'],
            context=raw[0]['context'], source_title=raw[0]['title'], revision=spec['revision'],
            turns=[dict(id=r['id'], question=r['question'], references=r['answers']['text']) for r in raw]))
    spec = config['tasks']['mrcr']
    needed = {row['source_row'] for row in lock['tasks']['mrcr']}
    raw_rows = {}
    stream = loader(spec['dataset'], spec['config'], split=spec['split'], revision=spec['revision'], streaming=True)
    for index, raw in enumerate(stream):
        if index in needed:
            raw_rows[index] = raw
        if index >= max(needed):
            break
    result['mrcr'] = []
    for expected in lock['tasks']['mrcr']:
        raw = raw_rows[expected['source_row']]
        context, query = flatten_mrcr(raw)
        tokens = len(tokenizer.encode(context))  # original MRCR loader convention
        require(tokens <= spec['context_token_limit'], 'Locked MRCR context exceeds limit; no truncation')
        result['mrcr'].append(dict(
            id=expected['id'], source_row=expected['source_row'], context_hash=digest(context),
            context=context, context_tokens=tokens, n_needles=raw['n_needles'], revision=spec['revision'],
            turns=[dict(id=expected['question_ids'][0], question=query, references=[raw['answer']],
                        random_string_to_prepend=raw['random_string_to_prepend'])]))
    spec = config['tasks']['scbench_summary']
    dataset = loader(spec['dataset'], spec['config'], split=spec['split'], revision=spec['revision'])
    require(len(dataset) == sum(config['scbench_split'][s] for s in ('train', 'validation', 'test')), 'SCBench source count changed')
    for rows in lock['scbench_splits'].values():
        for expected in rows:
            raw = dataset[expected['row']]
            require(str(raw['id']) == str(expected['source_id']) and digest(raw['context']) == expected['context_hash'], 'SCBench source/split hash changed')
    result['scbench_summary'] = []
    for expected in lock['tasks']['scbench_summary']:
        raw = dataset[expected['source_row']]
        result['scbench_summary'].append(dict(
            id=expected['id'], source_row=expected['source_row'], source_id=raw['id'],
            context_hash=digest(raw['context']), context=raw['context'], revision=spec['revision'],
            context_tokens=len(tokenizer.encode(raw['context'], add_special_tokens=False)),
            turns=[dict(id=f"scbench_{expected['source_row']}_{i}", question=t['input'], references=[t['answer']])
                   for i, t in enumerate(raw['multi_turns'])]))
    for task in TASKS:
        verify_rows(result[task], lock['tasks'][task])
    return result


def atomic_write(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'config.json')
    parser.add_argument('--ids', type=Path, default=HERE / 'ids.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--verify-existing', type=Path, help='Read-only check against completed full_*.jsonl run manifests')
    args = parser.parse_args()
    config, lock = load_spec(args.config, args.ids)
    if args.verify_existing:
        verify_existing(args.verify_existing, config, lock)
        print('Exact final evaluation selection and SCBench split verified')
        return
    if args.output_dir is None:
        parser.error('--output-dir or --verify-existing is required')
    marker = args.output_dir / 'complete.json'
    if marker.exists():
        saved = read(marker)
        require(saved['config_sha256'] == file_digest(args.config) and saved['ids_sha256'] == file_digest(args.ids), 'Incompatible selection')
        require(set(saved['files']) == set(OUTPUTS), 'Incomplete selection marker')
        require(all(file_digest(args.output_dir / n) == h for n, h in saved['files'].items()), 'Incomplete/corrupt selection')
        print('Existing selection verified:', args.output_dir)
        return
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()), 'Use an empty output directory; incomplete outputs are preserved')
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer']['id'], revision=config['tokenizer']['revision'])
    data = materialize(config, lock, tokenizer, load_dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        atomic_write(args.output_dir / f'{name}.jsonl', ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    atomic_write(args.output_dir / 'scbench_split_ids.json', json.dumps(lock['scbench_splits'], indent=2) + '\n')
    atomic_write(marker, json.dumps(dict(config_sha256=file_digest(args.config), ids_sha256=file_digest(args.ids),
                                        files={name: file_digest(args.output_dir / name) for name in OUTPUTS}), indent=2) + '\n')
    print('Selection verified and written:', args.output_dir)


if __name__ == '__main__':
    main()
