# Training and evaluating a Fast KVzip gate (Great Lakes)

All dependencies are managed by **mamba** through one file, `environment.yml`. It covers gate training, the prefill benchmarks and the math benchmarks.
The default model everywhere is `Qwen/Qwen3-4B-Instruct-2507`.

## 1. One-time environment setup

Run on the login node, from the repo root:

```bash
cd FastKVzipCL
bash setup_env.sh
```
Re-running `setup_env.sh` after editing `environment.yml` updates the env in place.

## 2. Activating the environment (any node)

```bash
cd FastKVzipCL && source env.sh
```
`env.sh` puts the HuggingFace and conda caches on scratch (`/scratch/cse595f26_class_root/cse595f26_class/$USER`), since the home quota is 80 GB.

## 3. Train a gate

Training has two steps. Run them as two Slurm jobs, the second only after the first succeeds:

```bash
mkdir -p logs
jid=$(sbatch --parsable train_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 feature)
sbatch --dependency=afterok:$jid --mem=480G train_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 optim
```

| Step      | What it does | Resources |
|-----------|--------------|-----------|
| `feature` | `prefill/feature.py -g ""`: prefills fineweb texts, scores the KV cache with KVzip (no gate), saves hidden states + scores | 128G RAM (default). ~170 GB of features for Qwen3-4B |
| `optim`   | `optim.py`: fits the gate on those features | loads every feature into RAM, so it needs `--mem=480G` |

Usage: `sbatch train_gate.sbatch [MODEL] [STEP]`, where STEP is `feature`, `optim` or `all` (default `all`).

- Features are written to scratch: `$FASTKVZIP_FEATURE_DIR/<model>` (default `/scratch/engin_root/engin1/$USER/fastkvzip_data`). `./data/<model>` is a symlink to it. Scratch is purged after ~60 days unused.
- The `feature` step resumes: samples whose `score_<idx>.pt` already exists are skipped. If it hits the 4 h limit, resubmit the same command.
- The gate and its training plots are saved to `result_gate/<model>/`:
  ```
  q<G>_dim16_sink16.pt          # the gate (G = query group size, 4 for Qwen3-4B)
  loss_q<G>_dim16_sink16.png    # training/test loss
  acc_q<G>_dim16_sink16.png     # training/test accuracy
  ```
- Without Slurm, `source train_gate.sh <model> <gpu>` runs both steps on one machine.

## 4. Give your gate a tag

A gate saved as `q4_dim16_sink16.pt` has the same name as the authors' released gate. While that file is there, a `fastkvzip` evaluation uses your gate instead of the released one. Rename it with a tag so you can run both:

```bash
cd result_gate/qwen3-4b-instruct-2507
for f in q4_dim16_sink16.pt acc_q4_dim16_sink16.png loss_q4_dim16_sink16.png; do
    mv -n "$f" "${f%.*}_mygate.${f##*.}"
done
```

Our current trained gate is `result_gate/qwen3-4b-instruct-2507/q4_dim16_sink16_mygate.pt`.
(You can also give the tag when training: `python -B optim.py -m <model> --tag mygate` saves `q4_dim16_sink16_mygate.pt` directly.)

How the loader finds a gate (`prefill/attention/gate.py`, `math/method/load_gate.py`):
1. `result_gate/<model>/<GATE>.pt`, or the same path under `$FASTKVZIP_GATE_DIR` if that is set.
2. Otherwise, it downloads `<GATE>.pt` from HuggingFace (`Jang-Hyun/Fast-KVzip`). Only `fastkvzip` exists there.

The log prints `Using local gate: ...` or `No local gate ... downloading`, so you can see which one ran.

## 5. Run the evaluation

From the repo root:
```bash
sbatch [--array=0-K] eval_gate.sbatch [MODEL] [DATA] [GATE] [NUM] [RUN]
```

| Arg   | Default                        | Meaning |
|-------|--------------------------------|---------|
| MODEL | `Qwen/Qwen3-4B-Instruct-2507`  | HF model id |
| DATA  | `all`                          | `all`, `qa`, `retv`, `redun`, or one dataset (`squad`, `gsm`, `scbench_kv`, `mrcr`, ...) |
| GATE  | `fastkvzip`                    | `fastkvzip` = released gate `q<G>_dim16_sink16`; or a local tag, e.g. `q4_dim16_sink16_mygate` |
| NUM   | `100`                          | max samples per dataset |
| RUN   | (none)                         | optional run tag; each tag gets its own results folder (e.g. `run1`) |

With `--array=0-K`, the NUM samples are split evenly across the K+1 tasks, so each task fits in the 4 h limit.

Examples:
```bash
# released gate on MRCR (100 samples in 4 tasks of 25)
sbatch --array=0-3 eval_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 mrcr fastkvzip 100 run1
# our trained gate on MRCR
sbatch --array=0-3 eval_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 mrcr q4_dim16_sink16_mygate 100
# quick test
sbatch eval_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 squad q4_dim16_sink16_mygate 10
```

- MRCR runs `prefill/eval_chunk_mrcr.py`. It uses the first NUM conversations (in dataset order) that fit in 128K tokens, so the same NUM always gives the same samples and runs are directly comparable.
- Other datasets run `prefill/eval_chunk.py`.
- Slurm settings: account `cse595f26_class`, partition `gpu_mig40` (half an A100, 40 GB), 1 GPU, 64G RAM, 4 h (the class maximum). For a bigger model, or if the MIG queue is long, use `--partition=spgpu` (A40, 48 GB). The GPU must be Ampere or Hopper; V100, P40 and RTX Pro 6000 (Blackwell) won't work.
- Logs go to `logs/fastkvzip-eval-<jobid>_<task>.out`.

## 6. Results

Each job generates answers with the full KV cache and with 75%, 50%, 40%, 30% and 20% of it kept, then scores them.
- Raw outputs: `prefill/results/<dataset>/<idx>_<method>/output-pair.json`
- Summary: `eval_results/<method>_<data>.json`

`<method>` is `<model>_<GATE>[_<RUN>]_chunk16k_w4096`, for example:
```
eval_results/qwen3-4b-instruct-2507_fastkvzip_run1_chunk16k_w4096_mrcr.json            # released gate
eval_results/qwen3-4b-instruct-2507_q4_dim16_sink16_mygate_chunk16k_w4096_mrcr.json    # our gate
```

```json
{
  "method": "qwen3-4b-instruct-2507_fastkvzip_run1_chunk16k_w4096",
  "level": "pair",
  "datasets": {
    "mrcr": {
      "num_samples": 100,
      "score":          {"1.0": 53.03, "0.75": 52.69, ...},
      "relative_score": {"1.0": 100.0, "0.75": 99.36, ...}
    }
  },
  "avg_relative_score": {"1.0": 100.0, "0.75": ..., ...}
}
```
- `score`: task score (%) at each KV ratio (fraction of the cache kept).
- `relative_score`: the score as a % of the full-cache score.
- `avg_relative_score`: the average across datasets (the paper's headline number; only for multi-dataset runs).

With `--array`, every task re-scores all samples finished so far. The JSON is complete once the last task finishes.

### Baseline: released gate, MRCR, 100 samples

| KV kept           | 1.0   | 0.75  | 0.5   | 0.4   | 0.3   | 0.2   |
|-------------------|-------|-------|-------|-------|-------|-------|
| score             | 53.03 | 52.69 | 51.44 | 52.95 | 50.80 | 32.79 |
| relative score    | 100   | 99.36 | 97.00 | 99.85 | 95.79 | 61.83 |

`run1` and `run2` gave the same numbers, so the evaluation is deterministic. The 1.0 column doesn't use the gate, so it should be the same for any gate.

To re-score existing outputs without regenerating them:
```bash
cd prefill
python -B -m results.parse -m qwen3-4b-instruct-2507_fastkvzip_chunk16k_w4096 -d all -o ../eval_results/out.json
python -B -m results.parse_mrcr -m qwen3-4b-instruct-2507_fastkvzip_run1_chunk16k_w4096 -n 100 -o ../eval_results/out_mrcr.json
```

## 7. Files added or changed

| File | Change |
|------|--------|
| `environment.yml` | new: all dependencies for the repo (replaces `prefill/requirements.txt` and `math/requirements.txt`, now deleted) |
| `setup_env.sh` | new: creates or updates the mamba env, then compiles the kernel |
| `build_kernel.sh` | new: compiles `csrc` (`tiny_api_cuda`) into the env; run on a compute node |
| `env.sh` | new: activates the env and puts caches on scratch; source it on any node |
| `train_gate.sbatch` | new: gate training job (`feature` / `optim` / `all`), features on scratch |
| `eval_gate.sbatch` | new: eval and scoring job; supports `--array` sample splitting and a RUN tag; runs with `sbatch` or `bash` |
| `prefill/attention/gate.py`, `math/method/load_gate.py` | local trained gate is now used first; the old fallback path was broken |
| `prefill/feature.py` | passes `-g` to the model, so `-g ""` scores with KVzip rather than a released gate; skips samples already saved (resume) |
| `prefill/data/load.py` | fix: fineweb indices are converted to plain ints (`datasets` rejects `numpy.int64`) |
| `prefill/eval_chunk_mrcr.py` | evaluates exactly samples `idx` to `idx+num`, in dataset order |
| `prefill/results/parse.py` | new `-o/--out` option writes scores to JSON |
| `prefill/results/parse_mrcr.py` | new `--idx`, `-n/--num` and `-o/--out` options |
| `optim.py`, `train_gate.sh`, `prefill/args.py`, `prefill/profiling.py`, `prefill/run.sh` | default model is now `Qwen/Qwen3-4B-Instruct-2507` |
| `README.md`, `math/README.md` | install instructions now point to mamba |
| `.gitignore` | ignores `logs/`, csrc build outputs and `data/*` (feature symlinks) |
| `result_gate/qwen3-4b-instruct-2507/q4_dim16_sink16_mygate.pt` | our trained gate (renamed from `q4_dim16_sink16.pt`; ignored by git) |
| `prefill/feature_hint.py`, `train_gate_hint.sbatch` | new: gate training data from a task-hint prompt (section 8) |
| `prefill/model/wrapper.py` | new `scoring_response` (scores context KV by attention from given tokens); `generate(..., return_ids=True)` |
| `prefill/attention/score.py` | scoring tokens can be fed in pieces (`score_tail_from`); unchanged when unset |
| `optim.py`, `load_data.py` | new `--folders` option to choose the feature folders (default: the two fineweb folders) |

## 8. Train a gate on a task-hint prompt

Instead of KVzip's "Repeat the previous context" prompt, `prefill/feature_hint.py` appends a hint saying what kind of question will come (it doesn't give the question itself). The model then generates a response, and each context KV is scored by its max attention from the hint and response tokens. Hidden states and scores are saved in the same format as `feature.py`:

```bash
jid=$(sbatch --parsable train_gate_hint.sbatch Qwen/Qwen3-4B-Instruct-2507 feature mrcr hint)
sbatch --dependency=afterok:$jid --mem=480G train_gate_hint.sbatch Qwen/Qwen3-4B-Instruct-2507 optim mrcr hint
# gate: result_gate/qwen3-4b-instruct-2507/q4_dim16_sink16_mrcr_hint.pt
sbatch --array=0-3 eval_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 mrcr q4_dim16_sink16_mrcr_hint 100
```

- Features go to `data/<model>/<DATA>_<TAG>/` (on scratch), next to the fineweb features. Nothing is overwritten.
- Hints are in `HINTS` in `feature_hint.py` (`mrcr`, `qa`, `retrieval_hint`). Pick one with `--hint qa` or pass any literal text through `FEATURE_ARGS="--hint '...'"`.
- Samples start at index 100 (`--start`), since the first 100 are used for evaluation. Contexts longer than `--max_ctx` (64K) are skipped. Collection stops at `--token_budget` (1M) context tokens, which is about what `optim` can load into 480 GB.
- `FOLDERS="mrcr_hint fineweb_10k:0-29 fineweb_10k_cat:0-5"` trains on the hint features plus the original fineweb features.
- Each run writes `responses.jsonl` (the generated responses) and `config.json` (the hint and settings) to the feature folder. Check a few responses to make sure the model follows the hint.

### `retrieval_hint`

Says only that questions requiring retrieval from the context will come, nothing about the task format. Asks for a short index of the context, one line per part with a few details in the model's own words (quoting made the model loop on one sentence).

```bash
jid1=$(sbatch --parsable --export=ALL,FEATURE_ARGS="--hint retrieval_hint --max_new_tokens 3072" train_gate_hint.sbatch Qwen/Qwen3-4B-Instruct-2507 feature mrcr retrieval_hint)
jid2=$(sbatch --parsable --dependency=afterok:$jid1 --mem=480G train_gate_hint.sbatch Qwen/Qwen3-4B-Instruct-2507 optim mrcr retrieval_hint)
sbatch --dependency=afterok:$jid2 --array=0-3 eval_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 mrcr q4_dim16_sink16_mrcr_retrieval_hint 100
```
