# method_list=('RWKV6' 'GLA' 'RetNet' 'Mamba' 'GatedDeltaNet' 'Mamba2' 'BIBS2' 'SSIB2')
method_list=('gla') # 'mamba2' 'ibm2')
# arc_easy,winogrande,boolq,piqa,gpqa_main_n_shot,mmlu,openbookqa,social_iqa,truthfulqa_mc1,truthfulqa_mc2

for i in "${!method_list[@]}"; do
    method="${method_list[$i]}"
    device="$i"  # 一个 method 对应一个 GPU
    
    model_path="output/${method}"
    log_file="logs/eval_${method}.log"
    
    echo "Launching $method on CUDA:$device, logging to $log_file"

    python eval_harness.py --model fla \
        --model_args pretrained=$model_path \
        --tasks truthfulqa_mc1,truthfulqa_mc2 \
        --batch_size 64 --device cuda:$device > "$log_file" 2>&1 &
done


