
JOBNAME=$(cat logs/debug/_jobname)
echo "Getting host for Slurm job $JOBNAME"

# Start TCP proxy from compute node
SLURM_COMPUTE_VM=$(squeue --me --name=$JOBNAME --states=R -h -O NodeList:100 | xargs)
echo "Starting ssh  from ${SLURM_COMPUTE_VM}:5000 to 127.0.0.1:$1"
ssh -L 127.0.0.1:$1:${SLURM_COMPUTE_VM}:5000 ${SLURM_COMPUTE_VM}

SSH_PID=$!

while kill -0 $SSH_PID 2>/dev/null; do
    sleep 1
done

echo "SSH tunnel closed, cancelling job"
scancel $SLURM_JOB_ID