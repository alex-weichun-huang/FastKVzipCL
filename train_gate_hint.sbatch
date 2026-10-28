#!/usr/bin/env bash
#SBATCH --job-name=fastkvzip-hint
#SBATCH --account=cse595f26_class
#SBATCH --partition=gpu_mig40     # 1/2 of an A100 80GB (MIG 3g.40gb, 40GB)
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                # feature step: CPU hidden states of a long context; override for optim
#SBATCH --time=04:00:00           # cse595f26_class max is 4h
#SBATCH --output=logs/%x-%j.out

# Train a gate on attention scores from a task-hint prompt (prefill/feature_hint.py)
# instead of KVzip's "repeat the context" prompt.
#   feature : prefill <DATA> contexts, generate a response to the hint, save hidden states + scores
#             to ./data/<model>/<DATA>_<TAG>/
#   optim   : fit the gate on those features, save result_gate/<model>/q<G>_dim16_sink16_<DATA>_<TAG>.pt
#
# Usage:  sbatch train_gate_hint.sbatch [MODEL] [STEP] [DATA] [TAG]
#   MODEL: HF model id            (default Qwen/Qwen3-4B-Instruct-2507)
#   STEP : feature | optim | all  (default all)
#   DATA : dataset name           (default mrcr)
#   TAG  : feature folder tag     (default hint)
# Optional environment variables:
#   FEATURE_ARGS : extra args for feature_hint.py, e.g. "--hint qa --token_budget 500000"
#   FOLDERS      : feature folders for optim (default "<DATA>_<TAG>"),
#                  e.g. "mrcr_hint fineweb_10k:0-29 fineweb_10k_cat:0-5" to mix in the repeat-prompt features
#
# Recommended: two jobs, optim only after feature succeeds:
#   jid=$(sbatch --parsable train_gate_hint.sbatch Qwen/Qwen3-4B-Instruct-2507 feature mrcr hint)
#   sbatch --dependency=afterok:$jid --mem=480G train_gate_hint.sbatch Qwen/Qwen3-4B-Instruct-2507 optim mrcr hint
# The feature step resumes, so resubmit it if it hits the 4h limit.

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
STEP="${2:-all}"
DATA="${3:-mrcr}"
TAG="${4:-hint}"
NAME="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"
FOLDERS="${FOLDERS:-${DATA}_${TAG}}"

# Find the repo root (see eval_gate.sbatch)
if [ -f "$(dirname "$0")/env.sh" ]; then
    REPO="$(cd "$(dirname "$0")" && pwd)"
else
    REPO="${SLURM_SUBMIT_DIR:?submit from the repo root}"
fi

set +u; source "$REPO/env.sh"; set -u
if [ "${CONDA_PREFIX:-}" != "$HOME/.conda/envs/fastkvzip" ]; then
    echo "ERROR: conda env 'fastkvzip' not active. Run 'bash setup_env.sh' from the repo root first." >&2
    exit 1
fi
echo "python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Point ./data/<model> at scratch (same as train_gate.sbatch)
FEATURE_DIR="${FASTKVZIP_FEATURE_DIR:-/scratch/engin_root/engin1/$USER/fastkvzip_data}"
mkdir -p "$FEATURE_DIR/$NAME" "$REPO/logs"
LINK="$REPO/data/$NAME"
if [ -e "$LINK" ] && [ ! -L "$LINK" ]; then
    echo "ERROR: $LINK is a real folder; move it to $FEATURE_DIR/$NAME first." >&2
    exit 1
fi
ln -sfn "$FEATURE_DIR/$NAME" "$LINK"
echo "features: $LINK -> $(readlink "$LINK")"

if [ "$STEP" = "feature" ] || [ "$STEP" = "all" ]; then
    cd "$REPO/prefill"
    # shellcheck disable=SC2086
    python -B feature_hint.py -m "$MODEL" -d "$DATA" --tag "$TAG" ${FEATURE_ARGS:-}
    du -sh "$FEATURE_DIR/$NAME/${DATA}_${TAG}"
fi

if [ "$STEP" = "optim" ] || [ "$STEP" = "all" ]; then
    cd "$REPO"
    # shellcheck disable=SC2086
    python -B optim.py -m "$MODEL" --folders $FOLDERS --tag "${DATA}_${TAG}"
    ls -la "$REPO/result_gate/$NAME"
fi

echo "Done."
