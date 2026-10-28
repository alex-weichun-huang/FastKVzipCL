# ==============================================================================
# Gate training data from a task-hint prompt (instead of KVzip's "repeat the context").
#
# For each context:
#   1. prefill the context and save its hidden states (same as feature.py)
#   2. append a hint prompt that says what kind of question will come, let the model generate a response
#   3. score each context KV by its max attention from the hint + response tokens
# Saves hidden_<i>.pt / score_<i>.pt in the same format as feature.py, so optim.py can train on them:
#   ../data/<model>/<data>_<tag>/
#
# Example (MRCR, run from prefill/):
#   python -B feature_hint.py -m Qwen/Qwen3-4B-Instruct-2507 -d mrcr --tag hint
#   cd .. && python -B optim.py -m Qwen/Qwen3-4B-Instruct-2507 --folders mrcr_hint --tag mrcr_hint
# ==============================================================================
import argparse
import json
import os

import torch
from model import ModelKVzip

from data import DataWrapper, load_dataset_all
from utils import TimeStamp

# Hint prompts. --hint takes one of these names or any literal text.
QA_HINT = (
    "You will be asked questions about the previous context. Before that, list the key "
    "facts, names, numbers and details that such questions are likely to ask about."
)
# Says only that retrieval questions will come, nothing about the task format.
# Asks for details in the model's own words: asking for quotes made it repeat one sentence over and over.
RETRIEVAL_HINT = (
    "You will be asked questions that require retrieving specific information from the previous "
    "context. Before that, go through the context from start to end and write a short index of it: "
    "one line per distinct part, in order, saying what it is and a few specific details, in your own "
    "words, that would help you find it again. Every line must describe a different part."
)
HINTS = {
    "qa": QA_HINT,
    "mrcr": QA_HINT,  # MRCR also uses the QA hint
    "retrieval_hint": RETRIEVAL_HINT,
}

# The first 100 samples of these datasets are used for evaluation (eval_gate.sbatch, NUM=100)
EVAL_DATASETS = ["mrcr", "squad", "gsm"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-m", "--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("-d", "--data", type=str, default="mrcr", help="dataset name (data/load.py)")
    p.add_argument("--tag", type=str, default="hint", help="output folder is <data>_<tag>")
    p.add_argument("--hint", type=str, default="", help=f"hint name {list(HINTS)} or literal text; default: mrcr for mrcr, else qa")
    p.add_argument("--start", type=int, default=100, help="first dataset index (skip eval samples)")
    p.add_argument("--num", type=int, default=300, help="max dataset samples to scan")
    p.add_argument("--max_ctx", type=int, default=64000, help="skip contexts longer than this (tokens)")
    p.add_argument("--token_budget", type=int, default=10**6, help="stop after this many context tokens in total (optim loads all into RAM, ~0.18 MB/token for Qwen3-4B)")
    p.add_argument("--max_new_tokens", type=int, default=1024, help="max response length")
    p.add_argument("--piece", type=int, default=256, help="scoring tokens per forward pass (lower if out of GPU memory)")
    p.add_argument("--prefill_chunk", type=int, default=16000)
    return p.parse_args()


def get_context(name, sample):
    """Context text of a sample. MRCR keeps the conversation in 'prompt' (see eval_chunk_mrcr.py)."""
    if name == "mrcr":
        return sample["prompt"]
    return sample["context"]


if __name__ == "__main__":
    args = parse_args()
    hint = HINTS.get(args.hint or ("mrcr" if args.data == "mrcr" else "qa"), args.hint)
    assert hint.strip(), "empty hint: pass a name from HINTS or literal text with --hint"
    if args.data in EVAL_DATASETS and args.start < 100:
        print(f"**Warning** samples {args.start}~99 of {args.data} are also used for evaluation")

    model = ModelKVzip(args.model, kv_type="retain", gate_path_or_name="")
    model.gen_kwargs["max_new_tokens"] = args.max_new_tokens

    dataset = load_dataset_all(args.data, model.tokenizer, n_data=args.start + args.num)
    if args.data == "mrcr":
        model.set_chat_template()  # same template as eval_chunk_mrcr.py
    else:
        DataWrapper(args.data, dataset, model)  # sets the dataset's chat template
    hint_ids = model.apply_template(hint)

    path = f"../data/{model.name}/{args.data}_{args.tag}"
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/config.json", "w") as f:
        json.dump(vars(args) | {"hint_text": hint}, f, indent=2)
    print(f"Hint: {hint}\nSaving to {path}")

    tt = TimeStamp(verbose=True)
    i, total = 0, 0  # i: output file index
    for idx in range(args.start, min(args.start + args.num, len(dataset))):
        if total >= args.token_budget:
            break
        sample = dataset[idx]
        ctx_ids = model.encode(get_context(args.data, sample))
        n_tok = ctx_ids.shape[1]
        if n_tok > args.max_ctx:
            continue
        total += n_tok

        tag = f"{i}"
        i += 1
        # resume: score is saved after hidden, so its presence means this sample is done
        if os.path.isfile(f"{path}/score_{tag}.pt"):
            print(f"skip {args.data}-{idx} -> {tag} (already saved)")
            continue

        kv = model.prefill(ctx_ids, prefill_chunk_size=args.prefill_chunk, save_hidden=True)
        hidden = torch.stack(kv.hidden_cache, dim=0).squeeze()
        hidden = hidden[..., kv.start_idx : kv.end_idx, :]
        kv.hidden_cache = []
        tt(f"Prefill {args.data}-{idx} ({n_tok} tokens)")

        response, a_ids = model.generate(hint_ids, kv=kv, return_ids=True)
        tt(f"Generate ({a_ids.shape[1]} tokens)")

        score = model.scoring_response(
            kv, torch.cat([hint_ids, a_ids], dim=1), piece_size=args.piece
        )
        tt("Score")

        torch.save(hidden, f"{path}/hidden_{tag}.pt")
        torch.save(score.cpu(), f"{path}/score_{tag}.pt")
        with open(f"{path}/responses.jsonl", "a") as f:
            f.write(json.dumps({"file": tag, "idx": idx, "n_tokens": n_tok, "response": response}) + "\n")
        print(score.shape, hidden.shape, f"total {total} tokens")
        print(f"[response {idx}] {response[:500]}\n")

        del kv, hidden, score
        torch.cuda.empty_cache()

    print(f"Done: {i} samples, {total} context tokens in {path}")
