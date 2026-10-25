#!/usr/bin/env bash
#SBATCH --job-name=fastkvzip-eval
#SBATCH --account=engin1          # or cse595f26_class
#SBATCH --partition=spgpu         # needs an Ampere/Hopper GPU (A40/A100/H100)
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Evaluate a trained Fast KVzip gate on the prefill benchmarks and save scores as JSON.
#
# Usage:  sbatch eval_gate.sbatch [MODEL] [DATA] [GATE] [NUM]
#   MODEL: HF model id                     (default Qwen/Qwen2.5-7B-Instruct-1M)
#   DATA : dataset or group: all/qa/retv/redun/squad/...  (default all)
#   GATE : "fastkvzip" = default trained gate q{G}_dim16_sink16,
#          or another tag saved by optim.py (e.g. q4_dim16_sink16_mytag)
#   NUM  : max samples per dataset         (default 100)
#
# The gate is loaded from ./result_gate/<model>/<tag>.pt if it exists,
# otherwise the released gate is downloaded from HuggingFace.

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct-1M}"
DATA="${2:-all}"
GATE="${3:-fastkvzip}"
NUM="${4:-100}"

# Find the repo root. Under sbatch the script runs from a copy, so use the submit dir;
# when run directly (e.g. `bash eval_gate.sbatch` on an interactive node) use the script's dir.
if [ -f "$(dirname "$0")/env.sh" ]; then
    REPO="$(cd "$(dirname "$0")" && pwd)"
else
    REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
fi

# Environment: mamba env 'fastkvzip' + HF cache on scratch. Create it once with: bash setup_env.sh
set +u; source "$REPO/env.sh"; set -u
if [ "${CONDA_PREFIX:-}" != "$HOME/.conda/envs/fastkvzip" ]; then
    echo "ERROR: conda env 'fastkvzip' not active. Run 'bash setup_env.sh' from the repo root first." >&2
    exit 1
fi
echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd "$REPO/prefill"
mkdir -p "$REPO/logs" "$REPO/eval_results"

# 1) Generate answers with the full KV cache and at each compression ratio
python -B eval_chunk.py -g "$GATE" -m "$MODEL" -d "$DATA" --num "$NUM"

# 2) Score them and write JSON
METHOD="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')_${GATE}_chunk16k_w4096"
OUT="$REPO/eval_results/${METHOD}_${DATA}.json"
python -B -m results.parse -m "$METHOD" -d "$DATA" -o "$OUT"

echo "Done. Scores: $OUT"
