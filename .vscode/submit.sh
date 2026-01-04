#!/bin/bash

JOB=$RANDOM
TEMP_FILE="/tmp/vscode-debug.$JOB.slurm"

echo "Temporary job file: $TEMP_FILE" # show file name
# echo "Current running file: $1" # show which script is being run

# Cancel job on Ctrl+C Keyboard Interrupt
JOBNAME="debug$JOB"
trap "scancel -n $JOBNAME; exit 130" SIGINT

cat >$TEMP_FILE << EOL
#!/bin/bash
#SBATCH --job-name=$JOBNAME       
#SBATCH -o logs/debug/$JOBNAME.log                               
#SBATCH -e logs/debug/$JOBNAME.log  
#SBATCH --get-user-env  
#SBATCH --exclusive
#SBATCH --partition=p4-main
#SBATCH --time=01:00:00
#SBATCH --exclude=p4-main-st-p4d24xlarge-16

python -m pip install debugpy
head_node=$(hostname)
CUDA_VISIBLE_DEVICES=0 python -m debugpy --wait-for-client --listen 0.0.0.0:5000 torchrun --nproc_per_node 2 --nnodes 1 \
        --rdzv_endpoint $head_node:29523 \
        --rdzv_id 10086 \
        --rdzv_backend c10d \
        pretrain.py \
        --deepspeed "ds_config.json" \
        --dataset_cache_dir "../../hf_datasets/SlimPajama-627B" \
        --output_dir "output/ibm2" \
        --config_name ibm2 \
        --resume_from_checkpoint true \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --bf16 True
EOL


 
# display the job submission file
echo " ********************************* "
echo "   Job submission file : $TEMP_FILE "
echo " ///////////////////////////////////////// "
cat $TEMP_FILE
echo " ///////////////////////////////////////// "

# submit the job
SLURM_SBATCH_FLAGS=$2
sbatch --get-user-env $SLURM_SBATCH_FLAGS $TEMP_FILE


echo 'Waiting for Slurm job to begin..'
while true; do
 export JOB_STATUS=$(squeue -n $JOBNAME --format="%.2t" | tail -n1 | xargs)
 echo "Job Status : $JOB_STATUS"
 if [ "$JOB_STATUS" == "R" ]; then
   echo "Job started!"
   break
 else
   sleep 2
   tput cuu 1
 fi
done

echo $JOBNAME > logs/debug/_jobname

ln -sf "$PWD/logs/debug/$JOBNAME.log" logs/debug/debug.new

# Give the script some time to install and start debugpy server
sleep 10
