#!/usr/bin/env bash
# GEM gate trained on 4 tasks in order (5000 / 4 = 1250 steps each, random init):
#   1. fineweb_10k:0-29 + fineweb_10k_cat:0-5 as one task (the local gate's KVzip repeat-prompt features)
#   2. fineweb_10k_qa_hint
#   3. fineweb_10k_summary_hint
#   4. fineweb_10k_retrieval_hint
# Saves result_gate/qwen3-4b-instruct-2507/q4_dim16_sink16_fineweb_10k_final.pt (+ acc_/loss_ plots);
# the earlier GEM gate q4_dim16_sink16_fineweb_10k_gem.pt is left untouched
# Needs ~410GB of features in RAM, hence --mem=640G.
# Run from the repo root:  bash train_final.sh
set -euo pipefail

FOLDERS="fineweb_10k:0-29+fineweb_10k_cat:0-5 fineweb_10k_qa_hint fineweb_10k_summary_hint fineweb_10k_retrieval_hint" \
OPTIM_SCRIPT=optim_gem.py \
sbatch --job-name=gem-final --mem=640G \
    train_gate_hint.sbatch Qwen/Qwen3-4B-Instruct-2507 optim fineweb_10k final
