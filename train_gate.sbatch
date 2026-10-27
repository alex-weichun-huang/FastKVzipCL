#!/usr/bin/env bash
#SBATCH --job-name=fastkvzip-train
#SBATCH --account=cse595f26_class
#SBATCH --partition=gpu_mig40     # 1/2 of an A100 80GB (MIG 3g.40gb, 40GB)
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                # feature step: ~100K-token texts need ~2x18GB of CPU hidden states; override for optim
#SBATCH --time=04:00:00           # cse595f26_class max is 4h
#SBATCH --output=logs/%x-%j.out

# Train a Fast KVzip gate in two steps (same as train_gate.sh, but on Slurm):
#   feature : prefill fineweb texts with KVzip scoring, save hidden states + scores
#   optim   : fit the gate on those features, save result_gate/<model>/q<G>_dim16_sink16.pt
#
# Usage:  sbatch train_gate.sbatch [MODEL] [STEP]
#   MODEL: HF model id                     (default Qwen/Qwen3-4B-Instruct-2507)
#   STEP : feature | optim | all           (default all)
#
# Recommended: run the steps as two jobs, optim only after feature succeeds.
# optim loads every feature into RAM at once (~175GB for Qwen3-4B, peak ~2-3x that):
#   jid=$(sbatch --parsable train_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 feature)
#   sbatch --dependency=afterok:$jid --mem=480G train_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 optim
#
# Features are large, so they live on scratch:
#   $FASTKVZIP_FEATURE_DIR/<model>   (default /scratch/engin_root/engin1/$USER/fastkvzip_data)
# and ./data/<model> is a symlink to it, so feature.py and optim.py need no changes.
# Scratch is purged after ~60 days unused; the trained gate itself is saved in ./result_gate.

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
STEP="${2:-all}"
NAME="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')"

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

# Point ./data/<model> at scratch
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
    python -B feature.py -m "$MODEL" -g ""
    du -sh "$FEATURE_DIR/$NAME"
fi

if [ "$STEP" = "optim" ] || [ "$STEP" = "all" ]; then
    cd "$REPO"
    python -B optim.py -m "$MODEL"
    ls -la "$REPO/result_gate/$NAME"
fi

echo "Done."
