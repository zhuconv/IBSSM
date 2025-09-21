#!/bin/bash
#SBATCH --output=/cusp-data-efa/peihaow/jz/IBSSM/logs/posttrain_%x_%j.log
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --partition=p4-main
#SBATCH --time=2-00:00:00
#SBATCH --job-name=mamba2

if [[ -z "$SLURM_JOB_NODELIST" ]]; then
    echo "Interactive mode: no SLURM_JOB_NODELIST"
    head_node=$(hostname)
    nodes=("$head_node")
    NNODES=1
else
    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    head_node=${nodes[0]}
    NNODES=${#nodes[@]}
fi

RDVZ_ID=$RANDOM
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# export MASTER_ADDR="p4-main-st-p4d24xlarge-3"
export MASTER_PORT=23567
export WANDB_MODE=offline 
export NCCL_DEBUG=WARN
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}

echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT} "RDVZ_ID:"$RDVZ_ID

model_path=mistralai/Mamba-Codestral-7B-v0.1
method=$SLURM_JOB_NAME
# method=$(echo "$model_path" | cut -d'/' -f2 | sed 's/-7B.*//')

# tiiuae/falcon-mamba-7b
# tiiuae/falcon-mamba-7b-instruct
# mistralai/Mamba-Codestral-7B-v0.1

srun torchrun --nnodes=${NNODES} --nproc_per_node=8 --rdzv_id=$RDVZ_ID --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} posttrain.py \
    --method $method \
    --hf_model_name_or_path $model_path \
    --data_dir ./data/crystal_chat/2048 \
    --output_dir ./output/${method}_7b \
    --n_chunks 7 \
    --per_device_batch_size 1 \
    --accumulate_grad_batches 1 # > log/gibs.log 2>&1 &