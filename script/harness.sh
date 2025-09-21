# method_list=('RWKV6' 'GLA' 'RetNet' 'Mamba' 'GatedDeltaNet' 'Mamba2' 'BIBS2' 'SSIB2')
method_list=('gla') # 'rwkv6' 'mamba' 'mamba2' 'gated_deltanet' 'retnet' 'ibm2')
method_list=('mamba2_7b/ckpt_3' 'ibm2_7b/ckpt_5')
# arc_easy,winogrande,boolq,piqa,gpqa_main_n_shot,mmlu,openbookqa,social_iqa,truthfulqa_mc1,truthfulqa_mc2

for i in "${!method_list[@]}"; do
    method="${method_list[$i]}"
    device="$i"  # 一个 method 对应一个 GPU
    
    model_path="output/${method}"
    method_log_name="${method//\//_}"
    log_file="logs/harness/eval_${method_log_name}.log"
    
    echo "Launching $method on CUDA:$device, logging to $log_file"

    CUDA_VISIBLE_DEVICES=$device python eval_harness.py --model fla \
        --model_args pretrained=$model_path,dtype=bfloat16 \
        --tasks arc_easy,arc_challenge,hellaswag,winogrande,boolq,piqa,gpqa_main_n_shot,mmlu,openbookqa,social_iqa,truthfulqa_mc1,truthfulqa_mc2 \
        --batch_size 32 --device cuda > "$log_file" 2>&1 &
done


