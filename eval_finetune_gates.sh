#!/usr/bin/env bash
# Evaluate every fine-tuned gate (from train_gate_finetune.sh) on mrcr, squad and scbench_summary (100 samples each).
# Scores go to eval_results/q4_dim16_sink16_<gate>_finetune_<dataset>.json
# Run from the repo root:  bash eval_finetune_gates.sh
set -euo pipefail

MODEL=Qwen/Qwen3-4B-Instruct-2507
NUM=100

# short name -> gate file in result_gate/qwen3-4b-instruct-2507/<file>.pt
declare -A GATES=(
    [local]=q4_dim16_sink16_local_finetune
    [qa_hint]=q4_dim16_sink16_fineweb_10k_qa_hint_finetune
    [summary_hint]=q4_dim16_sink16_fineweb_10k_summary_hint_finetune
    [retrieval_hint]=q4_dim16_sink16_fineweb_10k_retrieval_hint_finetune
    [mixhint]=q4_dim16_sink16_fineweb_10k_mixhint_finetune
    [gem]=q4_dim16_sink16_fineweb_10k_gem_finetune
)
ORDER=(local qa_hint summary_hint retrieval_hint mixhint gem)

# Same array split as eval_all_gates.sh, so each task fits in 4h
declare -A ARRAY=([squad]=0-0 [mrcr]=0-3 [scbench_summary]=0-9)
DATASETS=(squad mrcr scbench_summary)   # fastest first

for name in "${ORDER[@]}"; do
    [ -f "result_gate/qwen3-4b-instruct-2507/${GATES[$name]}.pt" ] || { echo "missing gate ${GATES[$name]}" >&2; exit 1; }
done

for data in "${DATASETS[@]}"; do
    for name in "${ORDER[@]}"; do
        out="q4_dim16_sink16_${name}_finetune_${data}"
        jid=$(OUT_NAME="$out" sbatch --parsable \
            --job-name="eval-ft-${name}-${data}" --array="${ARRAY[$data]}" \
            eval_gate.sbatch "$MODEL" "$data" "${GATES[$name]}" "$NUM")
        echo "${name}_finetune / $data: job $jid -> eval_results/$out.json"
    done
done
