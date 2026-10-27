#!/usr/bin/env bash

for MODEL in Qwen/Qwen3-4B-Instruct-2507; do # Qwen/Qwen2.5-7B-Instruct-1M Qwen/Qwen3-8B google/gemma-3-12b-it
    python -B eval_chunk.py -g fastkvzip -m $MODEL -d all  # FastKVzip
    python -B eval.py -g "" -m $MODEL -d all  # KVzip (not using a prefill-chunk)
    # python -B eval_chunk.py -g head -m $MODEL -d all  # DuoAttention (no head scores for Qwen3-4B in utils/head_score)
    python -B eval_chunk.py -g expect -m $MODEL -d all  # Expected Attention
    python -B eval_chunk.py -g snap -m $MODEL -d all  # SnapKV
done

# For MRCR, run
# python -B eval_chunk_mrcr.py -g fastkvzip -m $MODEL