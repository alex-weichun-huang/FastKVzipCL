#!/usr/bin/env bash
#SBATCH --job-name=fastkvzip-eval
#SBATCH --account=cse595f26_class
#SBATCH --partition=gpu_mig40     # smallest GPU that fits: 1/2 of an A100 80GB (MIG 3g.40gb, 40GB)
#SBATCH --gpus=1
# GPU choice for Qwen3-4B-Instruct-2507 (MRCR, contexts up to 128K tokens):
#   weights ~8GB + full KV cache at 128K ~19GB + activations ~3GB = ~30GB peak.
#   V100 (16GB) / P40 (24GB) are too small and lack FlashAttention-2 / bf16.
#   RTX Pro 6000 (Blackwell, sm_120) can't run the csrc kernel (built for sm_80/sm_90).
#   For a larger model, or if the MIG queue is long, use:  --partition=spgpu  (A40, 48GB)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00            # cse595f26_class max is 4h; split long runs with --array (below)
#SBATCH --output=logs/%x-%A_%a.out

# Evaluate a trained Fast KVzip gate on the prefill benchmarks and save scores as JSON.
#
# Usage:  sbatch [--array=0-K] eval_gate.sbatch [MODEL] [DATA] [GATE] [NUM] [RUN]
#   MODEL: HF model id                     (default Qwen/Qwen3-4B-Instruct-2507)
#   DATA : dataset or group: all/qa/retv/redun/squad/.../mrcr  (default all)
#   GATE : "fastkvzip" = default trained gate q{G}_dim16_sink16,
#          or another tag saved by optim.py (e.g. q4_dim16_sink16_mytag)
#   NUM  : max samples per dataset         (default 100)
#   RUN  : optional run tag; each tag writes to its own results folder (e.g. run1, run2)
#
# With --array=0-K the NUM samples are split evenly across the K+1 tasks, so each task
# fits in the time limit. Example (100 MRCR samples in 4 tasks of 25):
#   sbatch --array=0-3 eval_gate.sbatch Qwen/Qwen3-4B-Instruct-2507 mrcr fastkvzip 100 run1
#
# The gate is loaded from ./result_gate/<model>/<tag>.pt if it exists,
# otherwise the released gate is downloaded from HuggingFace.

set -euo pipefail

MODEL="${1:-Qwen/Qwen3-4B-Instruct-2507}"
DATA="${2:-all}"
GATE="${3:-fastkvzip}"
NUM="${4:-100}"
RUN="${5:-}"

# Split samples across array tasks: task t handles [IDX, IDX + COUNT)
IDX=0; COUNT="$NUM"
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    NTASK="${SLURM_ARRAY_TASK_COUNT:-1}"
    CHUNK=$(( (NUM + NTASK - 1) / NTASK ))
    IDX=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
    COUNT=$(( NUM - IDX < CHUNK ? NUM - IDX : CHUNK ))
    if [ "$COUNT" -le 0 ]; then echo "No samples for task $SLURM_ARRAY_TASK_ID"; exit 0; fi
fi
echo "samples: $IDX ~ $(( IDX + COUNT - 1 )) of $NUM, run tag: ${RUN:-none}"
TAG_ARGS=(); [ -n "$RUN" ] && TAG_ARGS=(--tag "$RUN")

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

METHOD="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')_${GATE}${RUN:+_$RUN}_chunk16k_w4096"
OUT="$REPO/eval_results/${METHOD}_${DATA}.json"

if [ "$DATA" = "mrcr" ]; then
    # MRCR has its own generate+grade script and parser.
    # Samples are the first NUM conversations (dataset order) that fit in 128K tokens.
    python -B eval_chunk_mrcr.py -g "$GATE" -m "$MODEL" --idx "$IDX" --num "$COUNT" "${TAG_ARGS[@]}"
    # Scores whatever samples are finished so far; the last array task to finish gives the full set
    python -B -m results.parse_mrcr -m "$METHOD" -n "$NUM" -o "$OUT"
else
    # 1) Generate answers with the full KV cache and at each compression ratio
    python -B eval_chunk.py -g "$GATE" -m "$MODEL" -d "$DATA" --idx "$IDX" --num "$COUNT" "${TAG_ARGS[@]}"
    # 2) Score them and write JSON
    python -B -m results.parse -m "$METHOD" -d "$DATA" -o "$OUT"
fi

echo "Done. Scores: $OUT"
