model="${1:-Qwen/Qwen3-4B-Instruct-2507}"
device="${2:-0}"

cd prefill
CUDA_VISIBLE_DEVICES=$device python -B feature.py -m $model -g ""

cd ..
CUDA_VISIBLE_DEVICES=$device python -B optim.py -m $model
