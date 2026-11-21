# FastKVzipCL

Model everywhere: `Qwen/Qwen3-4B-Instruct-2507` (query group size 4, so every gate is named `q4_dim16_sink16_<tag>`).

---

## 1. How to reproduce

### Environment (once, on the login node)
```bash
cd FastKVzipCL
bash setup_env.sh      # creates ~/.conda/envs/fastkvzip and compiles csrc
source env.sh          # activate on any node; HF and conda caches stay in $HOME
```

### Train the local gate (original Fast KVzip recipe, KVzip repeat-prompt features)
```bash
jid=$(sbatch --parsable train_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 feature)
sbatch --dependency=afterok:$jid --mem=480G train_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 optim
# then rename q4_dim16_sink16.pt -> q4_dim16_sink16_local.pt (otherwise it shadows the released gate name)
```

### Hint features and hint gates
```bash
M=Qwen/Qwen3-4B-Instruct-2507
# features (one job per hint; resubmit if it hits 4h, it resumes)
sbatch --export=ALL,FEATURE_ARGS="--hint qa_hint --start 100 --num 50 --token_budget 400000" \
       train_gate_hint.sbatch $M feature fineweb_10k qa_hint
#   retrieval_hint: --start 150 ; summary_hint: --start 200

# one gate per hint
sbatch --mem=480G train_gate_hint.sbatch $M optim fineweb_10k qa_hint      # -> ..._fineweb_10k_qa_hint.pt

H="fineweb_10k_qa_hint fineweb_10k_retrieval_hint fineweb_10k_summary_hint"
# mixhint (all 3 hints mixed into one training set)
FOLDERS="$H" sbatch --mem=480G train_gate_hint.sbatch $M optim fineweb_10k mixhint
# GEM (folder order = task order: qa -> retrieval -> summary)
FOLDERS="$H" OPTIM_SCRIPT=optim_gem.py sbatch --mem=480G train_gate_hint.sbatch $M optim fineweb_10k gem
```
- `optim` loads every feature into RAM: a single hint fits in 200G; mixhint, GEM and local need 480G.
- Features are written to `data/qwen3-4b-instruct-2507` (a symlink to scratch, ~72 GB per hint; scratch is purged after ~60 days unused).

### Fine-tune from the released gate
```bash
bash train_gate_finetune.sh        # 6 jobs (local, 3 hints, mixhint, gem), saves <gate>_finetune.pt
```

### Evaluate
```bash
bash eval_all_gates.sh             # gates x {squad, mrcr, scbench_summary}, 100 samples each
# single gate / dataset:
OUT_NAME=q4_dim16_sink16_gem_finetune_mrcr \
  sbatch --array=0-3 eval_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 mrcr q4_dim16_sink16_fineweb_10k_gem_finetune 100
```
- Setup: chunked prefill 16K, window 4096, pair-level scoring, KV ratios 1.0 / 0.75 / 0.5 / 0.4 / 0.3 / 0.2.
- `--array` splits the samples evenly over tasks (SQuAD ~15 s/sample, MRCR ~3.5 min, SCBench-summary ~9.5 min, hence array sizes 1 / 4 / 10). scbench_summary has only 70 samples, so the last tasks may be empty.
- Every task re-scores all samples finished so far, so the JSON in `eval_results/` is final only when the last task ends. Plot with `eval_results.ipynb`.
- The gate loader looks in `result_gate/<model>/<GATE>.pt` first, otherwise downloads the released gate from HF `Jang-Hyun/Fast-KVzip`. The log prints which one it used.

To re-score without regenerating:
```bash
cd prefill
python -B -m results.parse      -m <method> -d <data> -o ../eval_results/out.json
python -B -m results.parse_mrcr -m <method> -n 100    -o ../eval_results/out_mrcr.json
```

---

## 2. Changes compared to the original FastKVzip

### Method
- **Task-hint features** (`prefill/feature_hint.py`): instead of KVzip's "repeat the previous context" prompt, the context is prefilled, a hint describing the kind of upcoming question is appended (`qa_hint`, `retrieval_hint`, `summary_hint`; texts in `HINTS`), and the model generates a response. Each context KV is scored by its max attention from the hint and response tokens. Output format matches `feature.py`.
- **Multi-hint training**: `mixhint` mixes all hint feature folders into one training set; **GEM** (`optim_gem.py`) trains on hints as sequential tasks with episodic memory and a per-layer QP gradient projection (continual learning).
- **Fine-tuning** from the released Fast KVzip gate via `--init_gate fastkvzip` instead of random init.

### Code
| File | Change |
|---|---|
| `environment.yml`, `setup_env.sh`, `build_kernel.sh`, `env.sh` | mamba-managed env plus kernel build for Great Lakes |
| `train_gate.sbatch`, `eval_gate.sbatch` | Slurm jobs; eval supports `--array` sample splitting, a RUN tag, and `OUT_NAME` |
| `prefill/attention/gate.py`, `math/method/load_gate.py` | local trained gate is used first (the old fallback path was broken) |
| `prefill/feature.py` | `-g ""` scores with KVzip, not a released gate; resumes already-saved samples |
| `prefill/feature_hint.py`, `train_gate_hint.sbatch` | **new**: task-hint feature collection; `OPTIM_SCRIPT` / `OPTIM_ARGS` / `FOLDERS` passthrough |
| `prefill/model/wrapper.py`, `prefill/attention/score.py` | score context KVs by attention from given tokens, fed in pieces |
| `prefill/data/load.py` | fineweb loads at least `n_data` samples (so hint runs can start past idx 100); numpy-int index fix |
| `prefill/eval_chunk_mrcr.py`, `prefill/results/parse*.py` | exact `idx..idx+num` sample ranges; `-o` JSON output |
| `optim.py`, `load_data.py` | `--folders` to choose feature folders (`name` or `name:start-end`); `--init_gate` to fine-tune from an existing gate |
| `optim_gem.py` | **new**: GEM continual training, one task per folder |
| `eval_all_gates.sh`, `train_gate_finetune.sh` | **new**: batch launchers for the experiments above |
| `eval_results.ipynb` | **new**: plots score vs KV ratio for each dataset and gate |
