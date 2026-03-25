#!/bin/bash
#SBATCH --output=/cusp-data-efa/peihaow/jz/IBSSM/logs/pretrain_%x_%j.log
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1         # One task per GPU? per node
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --partition=p4-main
#SBATCH --time=1-00:00:00

# source ~/.bashrc
# conda activate ibm
# cd /cusp-data-efa/peihaow/jz/IBSSM

export LOGLEVEL=INFO
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID:-0}
export WANDB_MODE=disabled
export CUDA_LAUNCH_BLOCKING=1
export HF_TOKEN=${HF_TOKEN:?"Please set HF_TOKEN environment variable"}


# --- Detect nodes ---
if [[ -z "$SLURM_JOB_NODELIST" ]]; then
    echo "Interactive mode: no SLURM_JOB_NODELIST"
    head_node=$(hostname)
    head_node_ip=$(hostname -I | awk '{print $1}')
    nodes=("$head_node")
    NNODES=1
    command="torchrun"
    NGPUS=${NGPUS:-1}
    METHOD=${METHOD:-"ibm2"}
else
    nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
    head_node=${nodes[0]}
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
    NNODES=${#nodes[@]}
    command="srun torchrun"
    METHOD=$SLURM_JOB_NAME
fi
echo "Head Node IP: $head_node_ip; NNODES: $NNODES"

NGPUS=${NGPUS:-8}

### Hyperparameters
setting=${setting:-"20B/4k"}
# ==== batch size setting ====
if [[ $setting == *"4k"* ]]; then
  global_batch_size=512
  local_batch_size=$(( 512 / NNODES ))
  micro_batch_size=8   # = 4
fi
# ==== max_tokens ====
if [[ $setting == *"20B"* ]]; then
  max_tokens=$(( 100000000000 / 5 ))   # 20B = 2e10
elif [[ $setting == *"100B"* ]]; then
  max_tokens=100000000000              # 1e11
fi
# ==== final args ====
batch_size=$(( local_batch_size / NGPUS ))
gradient_accumulation_steps=$(( batch_size / micro_batch_size ))
max_steps=$(( max_tokens / (global_batch_size * 4096) )) 

# ../../hf_datasets/SlimPajama-627B
# "/vcc-data/peihaow/SlimPajama-627B"

# python -m debugpy --wait-for-client --listen 0.0.0.0:5000 -m torch.distributed.launch

${command} --nproc_per_node $NGPUS --nnodes $NNODES \
        --rdzv_endpoint $head_node_ip:29523 \
        --rdzv_id $RANDOM \
        --rdzv_backend c10d \
        pretrain.py \
        --deepspeed "ds_config.json" \
        --dataset_cache_dir "${DATASET:-HuggingFaceFW/fineweb-edu}" \
        --dataset_cached "${CACHED:-huggingface}" \
        --dataset_subset "${DATASET_SUBSET:-sample-10BT}" \
        --output_dir "output/${METHOD}" \
        --config_name ${METHOD} \
        --resume_from_checkpoint true \
        --per_device_train_batch_size $micro_batch_size \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --report_to none \
        --max_steps $max_steps \
        --context_len 4096 \
        --warmup_ratio 0.01 \
        --weight_decay 0.1 \
        --learning_rate 4e-4 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --lr_scheduler_type cosine_with_min_lr \
        --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
        --save_steps 1000 \
        --save_total_limit 2 \
        --logging_steps 50 \
        --do_train True \
        --do_predict True \
        --save_strategy "steps" \
        --gradient_checkpointing False \
        --bf16 True
