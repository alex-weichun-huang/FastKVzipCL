#!/usr/bin/env bash
# Evaluate every trained gate on mrcr, squad and scbench_summary (100 samples each).
# Scores go to eval_results/q4_dim16_sink16_<gate>_<dataset>.json
# Run from the repo root:  bash eval_all_gates.sh
set -euo pipefail

MODEL=Qwen/Qwen3-4B-Instruct-2507
NUM=100

# short name -> gate file in result_gate/qwen3-4b-instruct-2507/<file>.pt
declare -A GATES=(
    [local]=q4_dim16_sink16_local
    [summary_hint]=q4_dim16_sink16_fineweb_10k_summary_hint
    [retrieval_hint]=q4_dim16_sink16_fineweb_10k_retrieval_hint
    [mixhint]=q4_dim16_sink16_fineweb_10k_mixhint
    [gem]=q4_dim16_sink16_fineweb_10k_gem
)
ORDER=(local summary_hint retrieval_hint mixhint gem)

# Array size per dataset, so each task fits in 4h
# (measured: squad ~15s/sample, mrcr ~3.5min/sample, scbench_summary ~9.5min/sample)
declare -A ARRAY=([squad]=0-0 [mrcr]=0-3 [scbench_summary]=0-9)
DATASETS=(squad mrcr scbench_summary)   # fastest first

for data in "${DATASETS[@]}"; do
    for name in "${ORDER[@]}"; do
        jid=$(OUT_NAME="q4_dim16_sink16_${name}_${data}" sbatch --parsable \
            --job-name="eval-${name}-${data}" --array="${ARRAY[$data]}" \
            eval_gate.sbatch "$MODEL" "$data" "${GATES[$name]}" "$NUM")
        echo "$name / $data: job $jid -> eval_results/q4_dim16_sink16_${name}_${data}.json"
    done
done
