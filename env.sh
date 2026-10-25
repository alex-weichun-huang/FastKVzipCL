# Shared environment for Fast KVzip on Great Lakes. Source this, don't run it:
#   source env.sh
module load mamba/py3.13   # note: mamba/py3.12 on Great Lakes is broken
eval "$(conda shell.bash hook)"
conda activate "$HOME/.conda/envs/fastkvzip" 2>/dev/null || echo "env.sh: conda env 'fastkvzip' not found; run 'bash setup_env.sh' first" >&2

# Large caches go on scratch (home quota is 80 GB)
SCRATCH_DIR=/scratch/cse595f26_class_root/cse595f26_class/$USER
export HF_HOME="$SCRATCH_DIR/hf_cache"
export CONDA_PKGS_DIRS="$SCRATCH_DIR/conda_pkgs"
mkdir -p "$HF_HOME" "$CONDA_PKGS_DIRS"

# Compile CUDA extensions against the env's own CUDA toolkit
[ -n "${CONDA_PREFIX:-}" ] && export CUDA_HOME="$CONDA_PREFIX"
