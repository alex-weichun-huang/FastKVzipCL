#!/usr/bin/env bash
# One-time environment setup for Fast KVzip with mamba.
# Run from the repo root:  bash setup_env.sh
#   1. creates/updates the conda env 'fastkvzip' from environment.yml
#   2. compiles the csrc kernel (tiny_api_cuda) into it
# Login nodes kill memory-heavy work, so when run outside a Slurm job this script
# re-launches itself on a CPU compute node via srun (~15-30 min, output streams here).
# Set SLURM_ACCOUNT_BUILD to change the account used (default engin1).
set -eo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    exec srun --account="${SLURM_ACCOUNT_BUILD:-engin1}" --partition=standard \
        --cpus-per-task=8 --mem=32G --time=02:00:00 --job-name=fastkvzip-setup \
        bash "$REPO/setup_env.sh"
fi

source "$REPO/env.sh" 2>/dev/null   # mamba module + caches (env may not exist yet)

# CPU nodes have no GPU; tell the solver which CUDA driver level to target.
export CONDA_OVERRIDE_CUDA=12.6

# Explicit prefix in $HOME (otherwise mamba may pick the read-only module dir under /sw)
ENV_PREFIX="$HOME/.conda/envs/fastkvzip"
mkdir -p "$HOME/.conda/envs"
if [ -d "$ENV_PREFIX/conda-meta" ]; then
    mamba env update -p "$ENV_PREFIX" -f environment.yml --prune
else
    mamba env create -y -p "$ENV_PREFIX" -f environment.yml
fi

bash "$REPO/build_kernel.sh"

source "$REPO/env.sh"
python -c "import torch, flash_attn, tiny_api_cuda, transformers, nltk, tree_sitter_languages, latex2sympy2; print('torch', torch.__version__, '| flash_attn', flash_attn.__version__, '| transformers', transformers.__version__, '| all imports OK')"
echo "Environment ready: $CONDA_PREFIX  (activate with: source env.sh)"
