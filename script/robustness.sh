#!/usr/bin/env bash
set -euo pipefail

# where the eval script lives
EVAL_SCRIPT="eval_robustness.py"

# models (use the paths you want)
declare -A MODELS
MODELS["ibm2"]="output/ibm2_7b/ckpt_6"
MODELS["ibm2g"]="output/ibm2g_7b/ckpt_6"
# MODELS["mamba2"]="output/mamba2_7b/ckpt_6"
# MODELS["codestral"]="mistralai/Mamba-Codestral-7B-v0.1"

# MODELS=(
#   "ibm2"
#   "mamba2"
# )

DATASETS=(
  "wnli"
)
  # "sst2"


# attacks
ATTACKS=(
  "stresstest"
  "checklist"
  "deepwordbug"
  "textbugger"
)

  # "textbugger"
  # "bertattack"
  # "textfooler"

# log dir
LOGDIR="logs/robust"
mkdir -p "${LOGDIR}"

# GPU assignment: will assign each (model, attack) pair to a GPU 0..7 in order
# there must be exactly len(MODELS)*len(ATTACKS) <= 8 GPUs for this script as-is
TOTAL_TASKS=$(( ${#MODELS[@]} * ${#ATTACKS[@]} ))
if [ ${TOTAL_TASKS} -gt 8 ]; then
  echo "Error: TOTAL_TASKS=${TOTAL_TASKS} > 8. Adjust mapping or available GPUs." >&2
  exit 1
fi

task_idx=0
for model in "${!MODELS[@]}"; do
  for attack in "${ATTACKS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
      gpu=$(( task_idx ))   # maps to 0..7
      # baseline command (you can add --device or other args if your script supports them)
      cmd=( python -u "${EVAL_SCRIPT}" --model ${MODELS[$model]} --attack "${attack}" --dataset $dataset )

      logfile="${LOGDIR}/${model}_${attack}_${dataset}.log"
      echo "Launching: CUDA_VISIBLE_DEVICES=${gpu} ${cmd[*]}  -> ${logfile}"
      # export CUDA_VISIBLE_DEVICES for the single-call environment only
      # use nohup and background so all tasks start concurrently
      CUDA_VISIBLE_DEVICES="${gpu}" nohup "${cmd[@]}" > "${logfile}" 2>&1 &

      task_idx=$(( task_idx + 1 ))
    done
  done
done

echo "Use 'tail -f ${LOGDIR}/*.log' to watch logs, or 'ps aux | grep ${EVAL_SCRIPT}' to check processes."