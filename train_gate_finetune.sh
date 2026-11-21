#!/usr/bin/env bash
# Re-train every gate, fine-tuned from the released Fast-KVzip gate (q4_dim16_sink16 on HuggingFace)
# instead of random init. Same data and hyperparameters as the original runs; reuses existing features.
# Saves result_gate/<model>/<original gate name>_finetune.pt
# Run from the repo root:  bash train_gate_finetune.sh
set -euo pipefail

MODEL=Qwen/Qwen3-4B-Instruct-2507
INIT="--init_gate fastkvzip"

# train_gate_hint.sbatch saves q4_dim16_sink16_<DATA>_<TAG>.pt (DATA only names the gate in the optim step)
submit() {  # job-name mem folders script data tag
    local jid
    jid=$(FOLDERS="$3" OPTIM_SCRIPT="$4" OPTIM_ARGS="$INIT" sbatch --parsable \
        --job-name="$1" --mem="$2" train_gate_hint.sbatch "$MODEL" optim "$5" "$6")
    echo "$1: job $jid -> q4_dim16_sink16_${5}_${6}.pt"
}

HINTS="fineweb_10k_qa_hint fineweb_10k_retrieval_hint fineweb_10k_summary_hint"

# local: KVzip repeat-prompt features (optim.py default folders)
submit ft-local          480G "fineweb_10k:0-29 fineweb_10k_cat:0-5" optim.py     local       finetune
submit ft-qa_hint        200G fineweb_10k_qa_hint                    optim.py     fineweb_10k qa_hint_finetune
submit ft-retrieval_hint 200G fineweb_10k_retrieval_hint             optim.py     fineweb_10k retrieval_hint_finetune
submit ft-summary_hint   200G fineweb_10k_summary_hint               optim.py     fineweb_10k summary_hint_finetune
submit ft-mixhint        480G "$HINTS"                               optim.py     fineweb_10k mixhint_finetune
submit ft-gem            480G "$HINTS"                               optim_gem.py fineweb_10k gem_finetune
