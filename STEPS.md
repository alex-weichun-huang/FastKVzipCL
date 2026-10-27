# Evaluating a trained Fast KVzip gate (Great Lakes)

All dependencies are managed by **mamba** through one file, `environment.yml`. It covers gate training, the prefill benchmarks and the math benchmarks.

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

## 3. Put your trained gate where the code finds it

Training (`train_gate.sh` → `optim.py`) saves gates to:

```
result_gate/<model-name-lowercase>/q<G>_dim16_sink16.pt
```

For example: `result_gate/qwen3-4b-instruct-2507/q4_dim16_sink16.pt`

- The loader checks this folder first. To use a different folder, set `export FASTKVZIP_GATE_DIR=/path/to/result_gate`.
- If no local file exists, it downloads the authors' released gate from HuggingFace. The log line `Using local gate: ...` or `No local gate ... downloading` tells you which one ran.

## 4. Run the evaluation

From the repo root:
```bash
mkdir -p logs
sbatch eval_gate.sbatch [MODEL] [DATA] [GATE] [NUM]
# quick test:
sbatch eval_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 squad fastkvzip 10
# full benchmark:
sbatch eval_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 all fastkvzip 100
```

| Arg   | Default                        | Meaning |
|-------|--------------------------------|---------|
| MODEL | `Qwen/Qwen3-4B-Instruct-2507`  | HF model id |
| DATA  | `all`                          | `all`, `qa`, `retv`, `redun`, or one dataset (`squad`, `gsm`, `scbench_kv`, `mrcr`, ...). `mrcr` runs `eval_chunk_mrcr.py` on the first NUM conversations (dataset order) that fit in 128K tokens |
| GATE  | `fastkvzip`                    | `fastkvzip` uses `q<G>_dim16_sink16`; or another gate tag saved by `optim.py` |
| NUM   | `100`                          | max samples per dataset |

The job uses account `engin1`, partition `spgpu`, 1 GPU and 24 h. Edit the `#SBATCH` lines to change these. The GPU must be Ampere or Hopper (A40/A100/H100); V100s won't work. The job log goes to `logs/fastkvzip-eval-<jobid>.out`.


## 5. Results

`eval_gate.sbatch` runs two steps:
1. `prefill/eval_chunk.py` generates answers with the full KV cache and with 75%, 50%, 40%, 30% and 20% of it kept. The raw outputs go to `prefill/results/<dataset>/<idx>_<method>/output-pair.json`.
2. `prefill/results/parse.py -o ...` scores them and writes the summary JSON:

```
eval_results/<model>_<gate>_chunk16k_w4096_<data>.json
```

```json
{
  "method": "qwen3-4b-instruct-2507_fastkvzip_chunk16k_w4096",
  "level": "pair",
  "ratios": [1.0, 0.75, 0.5, 0.4, 0.3, 0.2],
  "datasets": {
    "squad": {
      "num_samples": 100,
      "score":          {"1.0": ..., "0.75": ..., ...},
      "relative_score": {"1.0": 100.0, "0.75": ..., ...}
    }
  },
  "avg_relative_score": {"1.0": 100.0, "0.75": ..., ...}
}
```
- `score`: task score (%) at each KV ratio (fraction of the cache kept).
- `relative_score`: the score as a % of the full-cache score.
- `avg_relative_score`: the average across datasets (the paper's headline number).

To re-score existing outputs without regenerating them:
```bash
cd prefill && python -B -m results.parse -m qwen3-4b-instruct-2507_fastkvzip_chunk16k_w4096 -d all -o ../eval_results/out.json
```

## 6. Files added or changed

| File | Change |
|------|--------|
| `environment.yml` | new: all dependencies for the repo (replaces `prefill/requirements.txt` and `math/requirements.txt`, now deleted) |
| `setup_env.sh` | new: creates or updates the mamba env, then compiles the kernel |
| `build_kernel.sh` | new: compiles `csrc` (`tiny_api_cuda`) into the env; run on a compute node |
| `env.sh` | new: activates the env and puts caches on scratch; source it on any node |
| `eval_gate.sbatch` | new: eval and scoring job; runs with `sbatch` or `bash` |
| `prefill/attention/gate.py`, `math/method/load_gate.py` | local trained gate is now used first; the old fallback path was broken |
| `prefill/results/parse.py` | new `-o/--out` option writes scores to JSON |
| `README.md`, `math/README.md` | install instructions now point to mamba |
| `.gitignore` | ignores `logs/` and csrc build outputs |