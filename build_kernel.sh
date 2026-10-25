#!/usr/bin/env bash
# Compile the csrc CUDA kernel (tiny_api_cuda) into the 'fastkvzip' conda env.
# Needs more memory than login nodes allow, so setup_env.sh runs it via srun on a CPU node.
# Targets sm_80 + sm_90 explicitly, so no GPU is needed.
set -eo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
source "$REPO/env.sh"

cd "$REPO/csrc"
rm -rf build dist *.egg-info
python build.py bdist_wheel
pip install --no-deps --force-reinstall dist/*.whl
rm -rf build *.egg-info

python -c "import torch, tiny_api_cuda; print('tiny_api_cuda OK (torch', torch.__version__, ')')"
