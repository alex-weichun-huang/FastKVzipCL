"""Small synthetic selection tests; no dataset downloads, GPU or model weights."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from select_samples import (HERE, OUTPUTS, digest, file_digest, flatten_mrcr, load_spec,
                    materialize, verify_rows)


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


def fixture():
    qa = [dict(id=f'q{i}', context='synthetic QA', title='title', question=f'question{i}',
               answers={'text': [f'answer{i}', 'alternative']}) for i in range(2)]
    mr = [dict(prompt=json.dumps([dict(role='user', content=f'message{i}'),
                                  dict(role='assistant', content='response'),
                                  dict(role='user', content='query')]), answer='prefix answer',
               random_string_to_prepend='prefix', n_needles=2) for i in range(2)]
    sc = [dict(id=i+1, context=f'synthetic paper{i}', multi_turns=[dict(input='question', answer='answer')]) for i in range(3)]
    config = {'tokenizer': {'id': 'unused', 'revision': 'pinned'},
              'scbench_split': dict(train=1, validation=1, test=1), 'tasks': {}}
    for name in ('squad', 'mrcr', 'scbench_summary'):
        config['tasks'][name] = dict(dataset=name, config='default', split='test', revision='pinned',
                                     contexts=1, questions=2 if name == 'squad' else 1)
    config['tasks']['mrcr']['context_token_limit'] = 1000
    splits = {name: [dict(row=i, source_id=i+1, context_hash=digest(sc[i]['context']),
                         context_tokens=len(Tokenizer().encode(sc[i]['context'])))]
              for i, name in enumerate(('train', 'validation', 'test'))}
    context, _ = flatten_mrcr(mr[1])
    tasks = dict(
        squad=[dict(id='qa', source_row=0, context_hash=digest(qa[0]['context']), question_ids=['q0', 'q1'])],
        mrcr=[dict(id='mr', source_row=1, context_hash=digest(context), context_tokens=len(Tokenizer().encode(context)),
                   n_needles=2, question_ids=['mr1'])],
        scbench_summary=[dict(id='sc', source_row=2, source_id=3, context_hash=digest(sc[2]['context']),
                              context_tokens=len(Tokenizer().encode(sc[2]['context'])), question_ids=['scbench_2_0'])])
    lock = dict(scbench_splits=splits, tasks=tasks)
    data = dict(squad=qa, mrcr=mr, scbench_summary=sc)
    return config, lock, data


def write_spec(directory, config, lock):
    ids = directory / 'ids.json'
    ids.write_text(json.dumps(lock))
    config = dict(config, lock_sha256=file_digest(ids))
    cfg = directory / 'config.json'
    cfg.write_text(json.dumps(config))
    return cfg, ids


class SelectionTests(unittest.TestCase):
    def test_actual_lock_counts_and_separation(self):
        config, lock = load_spec(HERE / 'config.json', HERE / 'ids.json')
        self.assertEqual([(config['tasks'][n]['contexts'], config['tasks'][n]['questions'])
                          for n in ('squad', 'mrcr', 'scbench_summary')], [(100, 1037), (100, 100), (14, 70)])
        self.assertEqual([len(lock['scbench_splits'][s]) for s in ('train', 'validation', 'test')], [49, 7, 14])
        self.assertNotIn('fineweb', lock)
        for rows in lock['tasks'].values():
            for row in rows:
                self.assertLessEqual(set(row), {'id', 'source_row', 'source_id', 'context_hash',
                                               'context_tokens', 'n_needles', 'question_ids'})

    def test_all_three_loaders_and_pinned_revisions(self):
        config, lock, data = fixture()
        calls = []
        def loader(name, *args, **kwargs):
            calls.append(kwargs)
            return data[name]
        result = materialize(config, lock, Tokenizer(), loader)
        self.assertEqual([t['id'] for t in result['squad'][0]['turns']], ['q0', 'q1'])
        self.assertEqual(result['squad'][0]['turns'][0]['references'], ['answer0', 'alternative'])
        self.assertTrue(all(c['revision'] == 'pinned' for c in calls))
        self.assertTrue(calls[1]['streaming'])

    def test_question_order_context_hash_and_token_limit_fail_closed(self):
        config, lock, data = fixture()
        result = materialize(config, lock, Tokenizer(), lambda name, *a, **kw: data[name])
        changed = copy.deepcopy(result['squad'])
        changed[0]['turns'].reverse()
        with self.assertRaisesRegex(ValueError, 'Question IDs/order'):
            verify_rows(changed, lock['tasks']['squad'])
        changed = copy.deepcopy(result['squad'])
        changed[0]['context'] += 'changed'
        with self.assertRaisesRegex(ValueError, 'Context content'):
            verify_rows(changed, lock['tasks']['squad'])
        config['tasks']['mrcr']['context_token_limit'] = 1
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            materialize(config, lock, Tokenizer(), lambda name, *a, **kw: data[name])

    def test_training_context_cannot_enter_final_selection(self):
        config, lock, _ = fixture()
        lock['tasks']['scbench_summary'][0]['context_hash'] = lock['scbench_splits']['train'][0]['context_hash']
        with tempfile.TemporaryDirectory() as temp:
            cfg, ids = write_spec(Path(temp), config, lock)
            with self.assertRaisesRegex(ValueError, 'Training/validation'):
                load_spec(cfg, ids)

    def test_formatter_matches_teammate_base(self):
        path = HERE.parent / 'prefill/data/load.py'
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'build_prompt_text')
        namespace = {'json': json}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        _, _, data = fixture()
        self.assertEqual(flatten_mrcr(data['mrcr'][0]), namespace['build_prompt_text'](data['mrcr'][0]))

    def test_reload_checks_every_file_without_importing_dataset_packages(self):
        config, lock, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            cfg, ids = write_spec(directory, config, lock)
            output = directory / 'outputs'
            output.mkdir()
            for name in OUTPUTS:
                (output / name).write_text('synthetic fixture\n')
            marker = dict(config_sha256=file_digest(cfg), ids_sha256=file_digest(ids),
                          files={n: file_digest(output / n) for n in OUTPUTS})
            (output / 'complete.json').write_text(json.dumps(marker))
            command = [sys.executable, str(HERE / 'select_samples.py'), '--config', str(cfg), '--ids', str(ids), '--output-dir', str(output)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            (output / OUTPUTS[0]).write_text('changed')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
