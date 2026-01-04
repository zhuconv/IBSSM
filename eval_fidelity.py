from torch.utils.data import Dataset, DataLoader
import csv, os
import glob
import fire
import torch
from torch import nn
from transformers import AutoTokenizer
from lit_gpt.model import Mamba2ForClassification, Config
import datasets
from lightning.fabric.loggers import TensorBoardLogger
from finetune import get_dataset, load_model
FLAG = 1

def compute_integrated_gradients(model, input_ids, labels, steps=10, null_id=2):
    torch.set_grad_enabled(True)
    # 获取嵌入层和输入嵌入
    embed_layer = model.get_input_embeddings()
    input_embeds = embed_layer(input_ids) # .detach().requires_grad_(True)
    baseline_ids = torch.full_like(input_ids, null_id)
    baseline_embeds = embed_layer(baseline_ids)

    
    # 创建基线（全零嵌入）
    # baseline = torch.zeros_like(input_embeds)
    
    # 生成插值路径
    scaled_embeds = [baseline_embeds + (i/steps)*(input_embeds - baseline_embeds) for i in range(0, steps+1)]
    
    # 存储梯度
    total_grads = 0
    
    for embed in scaled_embeds:
        # embed.requires_grad = True
        embed.retain_grad()
        
        # 前向传播
        logits = model(inputs_embeds=embed).float()
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # 计算梯度
        loss.backward(retain_graph=True)
        grad = embed.grad
        total_grads += grad.detach()
    
    # 计算平均梯度并积分
    avg_grad = total_grads / steps
    integrated_grad = (input_embeds - baseline_embeds) * avg_grad
    importance_scores = torch.sum(integrated_grad, dim=-1)
    
    torch.set_grad_enabled(False)
    return importance_scores

def compute_average_gradcam(model, input_ids, labels):
    torch.set_grad_enabled(True)
    # 存储各层激活和梯度
    activations = []
    gradients = []
    
    # 定义钩子函数
    def forward_hook(module, input, output):
        activations.append(output[0].detach())
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())
    
    # 注册钩子到每个transformer层
    hooks = []
    for layer in model.transformer.h:
        hooks.append(layer.register_forward_hook(forward_hook))
        hooks.append(layer.register_backward_hook(backward_hook))
    
    # 前向传播
    input_embeds = model.get_input_embeddings()(input_ids)
    logits = model(inputs_embeds=input_embeds).float()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    # 反向传播
    model.zero_grad()
    loss.backward()
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 处理奇偶数问题（每个层有前向和反向两个钩子）
    # print("activations, gradients shape", len(activations), len(gradients))
    # activations = activations[::2]  # 取前向钩子的输出
    # gradients = gradients[1::2]     # 取反向钩子的梯度
    gradients = gradients[::-1]  # 将逆序的反向梯度转为正序

    
    # 计算各层的Grad-CAM
    cam_per_layer = []
    for act, grad in zip(activations, gradients):
        # 计算通道权重（沿序列维度平均）
        alpha = grad.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        # 加权激活并沿通道维度求和
        cam = (act * alpha).sum(dim=-1)         # [batch, seq_len]
        cam = torch.relu(cam)                   # 应用ReLU
        cam_per_layer.append(cam)
    
    # 多层平均
    average_cam = torch.stack(cam_per_layer).mean(dim=0)
    torch.set_grad_enabled(False)
    return average_cam

def compute_gradient_x_input(model, input_ids, labels):
    torch.set_grad_enabled(True)
    input_embeds = model.get_input_embeddings()(input_ids)  # (batch, seq_len, emb_dim)
    # input_embeds.requires_grad_(True)
    input_embeds.retain_grad()
    
    # 前向传播
    logits = model(inputs_embeds=input_embeds).float()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    # 反向传播计算梯度
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    # 计算 gradient x input
    gradients = input_embeds.grad  # (batch, seq_len, emb_dim)
    importance_scores = torch.sum(gradients * input_embeds, dim=-1)  # (batch, seq_len)
    
    torch.set_grad_enabled(False)
    return importance_scores

def valid_acc_fidelity(model, val_loader, device, method, mode='negative', null_id=2, top_k=5):
    """
    mode: 'negative' 替换最重要的 top-k token
          'positive' 替换最不重要的 top-k token
          'neutral'  计算 negative_acc_diff - positive_acc_diff
    top_k: 支持单个整数或整数列表
    """
    model.eval()
    # 统一处理为列表
    top_k_list = [top_k] if isinstance(top_k, int) else top_k
    
    # 初始化结果存储
    results = {
        k: {
            'total_correct': 0,
            'total_correct_modified_neg': 0,
            'total_correct_modified_pos': 0,
            'total_correct_modified': 0,
            'total_samples': 0
        } for k in top_k_list
    }

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 公共计算部分（所有k共享）
            # ========== 计算重要性分数 ==========
            if method == 'self':
                token_saliencys = []
                def get_token_saliency(module):
                    if hasattr(module, "get_token_saliency"):
                        saliency = module.get_token_saliency()
                        if saliency is not None:
                            token_saliencys.append(saliency)
                logits = model(input_ids).float()
                model.apply(get_token_saliency)
                scores = torch.stack(token_saliencys).mean(dim=0)
                
            elif method == 'gradx':
                logits = model(input_ids).float()
                scores = compute_gradient_x_input(model, input_ids, labels)
                
            elif method == 'ig':
                logits = model(input_ids).float()
                scores = compute_integrated_gradients(model, input_ids, labels, null_id=null_id)
                # print(scores)
            
            elif method == 'gradcam':
                logits = model(input_ids).float()
                scores = compute_average_gradcam(model, input_ids, labels)
                
            else:
                raise ValueError("Unsupported method")

            # 公共掩码和原始正确数
            mask = (input_ids != null_id)
            scores_clone = scores.clone()
            preds = torch.argmax(logits, dim=1)
            batch_correct = (preds == labels).sum().item()
            batch_samples = labels.size(0)

            # 对每个k值独立处理
            for k in top_k_list:
                # 克隆分数避免污染原始数据
                k_scores = scores_clone.clone()
                k_results = results[k]
                
                # 累计原始正确数
                k_results['total_correct'] += batch_correct
                k_results['total_samples'] += batch_samples

                # ========== 处理不同模式 ==========
                if mode == 'neutral':
                    # Negative模式处理
                    neg_scores = k_scores.clone()
                    neg_scores[~mask] = -float('inf')
                    _, neg_indices = torch.topk(neg_scores, k=k, dim=1, largest=True)
                    mod_neg = input_ids.scatter(1, neg_indices, null_id)
                    preds_neg = torch.argmax(model(mod_neg), dim=1)
                    k_results['total_correct_modified_neg'] += (preds_neg == labels).sum().item()

                    # Positive模式处理
                    pos_scores = k_scores.clone()
                    pos_scores[~mask] = float('inf')
                    _, pos_indices = torch.topk(pos_scores, k=k, dim=1, largest=False)
                    mod_pos = input_ids.scatter(1, pos_indices, null_id)
                    preds_pos = torch.argmax(model(mod_pos), dim=1)
                    k_results['total_correct_modified_pos'] += (preds_pos == labels).sum().item()

                else:  # 单模式处理
                    mode_scores = k_scores.clone()
                    mode_scores[~mask] = -float('inf') if mode == 'negative' else float('inf')
                    _, indices = torch.topk(mode_scores, k=k, dim=1, largest=(mode == 'negative'))
                    mod_input = input_ids.scatter(1, indices, null_id)
                    preds_mod = torch.argmax(model(mod_input), dim=1)
                    k_results['total_correct_modified'] += (preds_mod == labels).sum().item()


    # ========== 结果计算和输出 ==========
    acc_diffs = []
    for k in top_k_list:
        res = results[k]
        val_acc = res['total_correct'] / res['total_samples']
        global FLAG
        if FLAG:
            print(f"Original Acc: {val_acc:.4f}")
            FLAG = 0
        
        if mode == 'neutral':
            acc_neg = res['total_correct_modified_neg'] / res['total_samples']
            acc_pos = res['total_correct_modified_pos'] / res['total_samples']
            diff = (val_acc - acc_neg) - (val_acc - acc_pos)
            print(f"{method} | k={k} | Neutral Fidelity: {diff:.4f}({((diff + 1) * 50):.2f}%)")
            acc_diffs.append(diff)
        else:
            acc_mod = res['total_correct_modified'] / res['total_samples']
            diff = val_acc - acc_mod
            print(f"{method} | k={k} | {mode} Fidelity: {diff:.4f}({((diff + 1) * 50):.2f}%)")
            acc_diffs.append(diff)
    


    # 计算并打印平均结果（当有多个k值时）
    if len(top_k_list) > 1:
        avg_diff = sum(acc_diffs) / len(acc_diffs)
        print(f"{method} | Average Fidelity (k={top_k_list}): {avg_diff:.4f}({((avg_diff + 1) * 50):.2f}%)")
        return (avg_diff + 1) * 50
    
    return (acc_diffs[0] + 1) * 50

def v1_valid_acc_diff(model, val_loader, device, method, mode='negative', null_id=2, top_k=5):
    """
    mode: 'negative' 替换最重要的 top-k token
          'positive' 替换最不重要的 top-k token
          'neutral'  计算 negative_acc_diff - positive_acc_diff
    """
    model.eval()
    total_correct = 0
    total_correct_modified_neg = 0  # 仅用于 neutral 模式
    total_correct_modified_pos = 0  # 仅用于 neutral 模式
    total_correct_modified = 0      # 用于 non-neutral 模式
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 根据方法计算重要性分数（所有模式共享）
            if method == 'self':
                token_saliencys = []
                def get_token_saliency(module):
                    if hasattr(module, "get_token_saliency"):
                        saliency = module.get_token_saliency()
                        if saliency is not None:
                            token_saliencys.append(saliency)
                logits = model(input_ids).float()
                model.apply(get_token_saliency)
                scores = torch.stack(token_saliencys).mean(dim=0)
                
            elif method == 'gradx':
                logits = model(input_ids).float()
                scores = compute_gradient_x_input(model, input_ids, labels)
                
            elif method == 'ig':
                logits = model(input_ids).float()
                scores = compute_integrated_gradients(model, input_ids, labels, null_id=null_id)
            
            elif method == 'gradcam':
                logits = model(input_ids).float()
                scores = compute_average_gradcam(model, input_ids, labels)
                
            else:
                raise ValueError("Unsupported method")

            # 公共掩码逻辑
            mask = (input_ids != null_id)
            scores_clone = scores.clone()

            # 处理不同模式
            if mode == 'neutral':
                ############## 同时处理 negative 和 positive 模式 ##############
                # --- Negative 模式 ---
                masked_scores_neg = scores_clone.clone()
                masked_scores_neg[~mask] = -float('inf')
                _, topk_indices_neg = torch.topk(masked_scores_neg, k=top_k, dim=1, largest=True)
                modified_input_neg = input_ids.clone()
                modified_input_neg.scatter_(1, topk_indices_neg.to(device), null_id)
                logits_neg = model(modified_input_neg).float()
                preds_neg = torch.argmax(logits_neg, dim=1)

                # --- Positive 模式 ---
                masked_scores_pos = scores_clone.clone()
                masked_scores_pos[~mask] = float('inf')
                _, topk_indices_pos = torch.topk(masked_scores_pos, k=top_k, dim=1, largest=False)
                modified_input_pos = input_ids.clone()
                modified_input_pos.scatter_(1, topk_indices_pos.to(device), null_id)
                logits_pos = model(modified_input_pos).float()
                preds_pos = torch.argmax(logits_pos, dim=1)

                # 统计结果
                total_correct_modified_neg += (preds_neg == labels).sum().item()
                total_correct_modified_pos += (preds_pos == labels).sum().item()

            else:
                ############## 原始单模式逻辑 ##############
                masked_scores = scores_clone.clone()
                if mode == 'negative':
                    masked_scores[~mask] = -float('inf')
                    largest = True
                elif mode == 'positive':
                    masked_scores[~mask] = float('inf')
                    largest = False
                else:
                    raise ValueError(f"Invalid mode: {mode}")

                _, topk_indices = torch.topk(masked_scores, k=top_k, dim=1, largest=largest)
                modified_input_ids = input_ids.clone()
                modified_input_ids.scatter_(1, topk_indices.to(device), null_id)
                logits_modified = model(modified_input_ids).float()
                preds_modified = torch.argmax(logits_modified, dim=1)
                total_correct_modified += (preds_modified == labels).sum().item()

            # 公共原始正确数统计
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    # 计算最终指标
    val_acc = total_correct / total_samples

    if mode == 'neutral':
        modified_acc_neg = total_correct_modified_neg / total_samples
        modified_acc_pos = total_correct_modified_pos / total_samples
        acc_diff_neg = val_acc - modified_acc_neg
        acc_diff_pos = val_acc - modified_acc_pos
        acc_diff = acc_diff_neg - acc_diff_pos

        global FLAG
        if FLAG:
            print(f"Original Acc: {val_acc:.4f}")
            FLAG = 0
        print(f"{method.capitalize()} (neutral mode) Fidelity: ({acc_diff_neg:.4f} - {acc_diff_pos:.4f}) = {acc_diff:.4f} ({((acc_diff + 1) * 50):.2f}%)")
    else:
        modified_val_acc = total_correct_modified / total_samples
        acc_diff = val_acc - modified_val_acc
        print(f"{method.capitalize()} ({mode} mode) Fidelity Diff: {val_acc:.4f} - {modified_val_acc:.4f} = {acc_diff:.4f}")

    return acc_diff

def main(config_name, dataset_name, device='cuda', add_cls_token=True):
    # Load components
    checkpoint_pattern = f"finetune_checkpoints/{config_name}_{dataset_name}_epoch*.pth"
    # 使用 glob 找到所有匹配的文件路径
    checkpoint_path = glob.glob(checkpoint_pattern)[-1]

    tokenizer = AutoTokenizer.from_pretrained("Orkhan/llama-2-7b-absa", trust_remote_code=True)
    # train_dataset = get_dataset(dataset_name, tokenizer, add_cls_token=add_cls_token, split='train')
    
    val_dataset = get_dataset(dataset_name, tokenizer, add_cls_token=add_cls_token, split='validation')
    print("Length of valid dataset: ", len(val_dataset))
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Initialize model
    model = load_model(checkpoint_path, config_name, val_dataset.num_classes, device=device)

    mode = 'neutral'
    csv_file = "explanation/fidelity.csv"
    for method in ['self', 'gradx', 'gradcam', 'ig']: 
    # for method in ['ig']:
        # for top_k in [1, 3, 5, 7, 9]:
        fidelity = valid_acc_fidelity(model, val_loader, device, method=method, mode=mode, top_k=[1, 3, 5, 7, 9] if dataset_name != 'imdb' else [5, 10, 15, 20, 25])
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["explanation", "dataset", "model", "value"])
            
            writer.writerow([
                method,          # 解释方法
                dataset_name,         # 数据集名称
                config_name,      # 模型配置名称
                f"{fidelity:.2f}"  # 格式化后的指标值
            ])


if __name__ == '__main__':
    # CUDA_VISIABLE_DEVICES=1 python eval_fidelity.py --config_name GIBS2 --dataset_name emotion
    fire.Fire(main) # emotion, snli, sst2, medical-bios