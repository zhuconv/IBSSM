from datetime import datetime
from pytz import timezone
import time
from functools import partial
import wandb
import os
import fire
import json
import tqdm
import torch
import deepspeed
from fla.models import *
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import lightning as L
from lightning.fabric.strategies import FSDPStrategy, DeepSpeedStrategy
# from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import AutoTokenizer, AutoConfig
from pytorch_lightning.loggers import WandbLogger

from utils import (
    load_jsonl_examples,
    get_cosine_lr_decay_fn,
    get_grad_norm,
    save_checkpoint,
    get_last_ckpt_idx)


TIMEZONE = timezone('EST')
DATE = str(datetime.now(tz=TIMEZONE)).split()[0]
PROJECT_NAME = f''
RUN_NAME = f'stacking_{DATE}'

LEARNING_RATE = 3e-5
END_LEARNING_RATE = 0.
WARMUP_RATE = 0.1
GRAD_NORM_CLIP = 1.
WEIGHT_DECAY = 0.1
BETA1 = 0.9
BETA2 = 0.95
LOSS_SCALER = 0.1
ACCELERATOR = 'cuda'
PRECISION = 'bf16-mixed'
RANDOM_SEED = 11111


def collate_fn(examples, device):
    token_ids = torch.tensor(
        [example['token_ids'] for example in examples], device=device)
    tgt_mask = torch.tensor(
        [example['tgt_mask'] for example in examples], device=device)

    return {
        'input_ids': token_ids[:, :-1],
        'labels': token_ids[:, 1:],
        'loss_mask': tgt_mask[:, 1:]
    }


def train_epoch(fabric,
                output_dir,
                tokenizer,
                model,
                optimizer,
                lr_schedule_fn,
                examples,
                per_device_batch_size,
                accumulate_grad_batches,
                chunk_idx,
                run_wandb):
    step = chunk_idx * (len(examples) // per_device_batch_size)

    if fabric.global_rank == 0:
        output_file = open(f'{output_dir}/log_chunk_{chunk_idx}.jsonl', 'w')
                    
    example_batch_idxes = tqdm.trange(
        0, len(examples), per_device_batch_size,
        desc=f'Training chunk {chunk_idx} (gbs='
             f'{per_device_batch_size * fabric.world_size * accumulate_grad_batches})',
        disable=(fabric.global_rank > 0),
        leave=True,   # 保留进度条
        dynamic_ncols=True 
        )
    
    
    for i in example_batch_idxes:
        t0 = time.time()

        lr = lr_schedule_fn(step)
        step += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        is_accumulating = (step % accumulate_grad_batches != 0)

        batch = collate_fn(
            examples=examples[i:i+per_device_batch_size], device=fabric.device)
        
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids=batch['input_ids']).logits
            ce_loss = torch.nn.functional.cross_entropy(
                logits.reshape((-1, logits.size(-1))),
                batch['labels'].reshape(-1),
                reduction='none')
            
            #! HACK for IBS
            auxiliary_loss = []
            def collect_auxiliary_loss(module):
                if hasattr(module, "get_auxiliary_loss"):
                    val = module.get_auxiliary_loss()
                    auxiliary_loss.append(val)
            model.apply(collect_auxiliary_loss)
            tensors = [x for x in auxiliary_loss if isinstance(x, torch.Tensor)]

            # 求 mean，保持 [1, 2047] shape
            if tensors:
                kl_loss = torch.mean(torch.stack(tensors, dim=0), dim=0) * LOSS_SCALER
                loss = ce_loss + kl_loss
                kl_loss = kl_loss.mean() # just for logging
            else:
                loss = ce_loss
                kl_loss = 0

            loss = torch.sum(loss * batch['loss_mask'].reshape(-1)) / (
                    torch.sum(batch['loss_mask'].reshape(-1)) + 1e-5)

            fabric.backward(loss / accumulate_grad_batches)

        if not is_accumulating:
            grad_norm = get_grad_norm(model=model)
            fabric.clip_gradients(model, optimizer, max_norm=GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        log = {
            'loss': f"{loss.item():.4f}",
            'kl_loss': f"{kl_loss.item():.4f}" if kl_loss else 0,
            'lr': lr,
            'step': step,
            'speed(#tok/s/gpu)': int(
                batch['input_ids'].numel() / (time.time() - t0)),
        }
        if not is_accumulating:
            log.update({'grad_norm': grad_norm})

        example_batch_idxes.set_postfix(log)
        if run_wandb and fabric.global_rank == 0:
            wandb.log(log)
        if fabric.global_rank == 0:
            output_file.write(json.dumps(log) + '\n')
            output_file.flush()

    save_checkpoint(
        fabric=fabric,
        tokenizer=tokenizer,
        model=model,
        optimizer=optimizer,
        save_dir=f'{output_dir}/ckpt_{chunk_idx}')


def main(hf_model_name_or_path,
         output_dir,
         n_chunks,
         data_dir='crystal_chat/chunk',
         per_device_batch_size=10,
         accumulate_grad_batches=1,
         method: str='',
         run_wandb: bool=False,
         no_slurm: bool=True,
    ):

    with open(f'{data_dir}/chunk-{0}.jsonl') as f:
        n_samples_per_chunk = sum(1 for _ in f)
    print(f"*** {n_samples_per_chunk=} ***")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_name_or_path, trust_remote_code=True)

    if method in ['gamma', 'bernoulli', 'ibm2']:
        # load mamba2 model but initialize as ibm2
        from fla.models.ibm2.modeling_ibm2 import IBM2Block as Block # for FSDP auto wrap
        config = IBM2Config.from_pretrained(hf_model_name_or_path)
        config.ib_type = method
        model = IBM2ForCausalLM.from_pretrained(hf_model_name_or_path, config=config)
    elif method == 'mamba2':
        # or method == 'mamba-codestral':
        from fla.models.mamba2.modeling_mamba2 import Mamba2Block as Block
        config = Mamba2Config.from_pretrained(hf_model_name_or_path)
        model = Mamba2ForCausalLM.from_pretrained(hf_model_name_or_path, config=config)
    else:
        raise NotImplementedError
        from transformers import AutoModelForCausalLM
        import importlib
        assert method in ['rwkv6', 'gla', 'mamba'], f"not support metho {method}"
        capitalized_method = method.upper() if method != 'mamba' else 'Mamba'
        module = importlib.import_module(f"fla.models.{method}.modeling_{method}")
        Block = getattr(module, f"{capitalized_method}Block")
        # from transformers import RwkvForCausalLM
        model = AutoModelForCausalLM.from_pretrained(hf_model_name_or_path, use_cache=False, trust_remote_code=True) #.to(dtype=torch.bfloat16)

    if no_slurm:
        ENV_2_REMOVE = [k for k in os.environ.keys() if k.startswith('SLURM_')]
        for ENV_NAME in ENV_2_REMOVE:
            os.environ.pop(ENV_NAME)
    
    
    devices = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    nnodes = int(os.environ.get("WORLD_SIZE", 1)) // devices
    fabric = L.Fabric(
        accelerator=ACCELERATOR,
        num_nodes=nnodes,
        devices=devices,
        precision=PRECISION,
        strategy=FSDPStrategy(
            state_dict_type='full', # 'sharded'
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={Block}),
            activation_checkpointing_policy={Block},
            cpu_offload=True,
            limit_all_gathers=True
            ),
        # DeepSpeedStrategy(config="ds_stage3.json"),
    )
    fabric.launch()

    if fabric.global_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        if run_wandb:
            wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    last_ckpt_idx = get_last_ckpt_idx(workdir=output_dir)
    fabric.seed_everything(RANDOM_SEED + last_ckpt_idx + 1)

    # torch.optim.AdamW 
    # deepspeed.ops.adam.DeepSpeedCPUAdam
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(BETA1, BETA2),
        foreach=False
        )

    model, optimizer = fabric.setup(model, optimizer)
    if last_ckpt_idx != -1:
        fabric.load(
            path=f'{output_dir}/ckpt_{last_ckpt_idx}/fabric_ckpt',
            state={'model': model, 'optimizer': optimizer})

    global_micro_batch_size = per_device_batch_size * fabric.world_size
    total_steps = n_samples_per_chunk // global_micro_batch_size * n_chunks
    warmup_steps = int(total_steps * WARMUP_RATE)
    lr_schedule_fn = get_cosine_lr_decay_fn(
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        learning_rate=LEARNING_RATE,
        end_learning_rate=END_LEARNING_RATE)

    model.train()
    for chunk_idx in range(last_ckpt_idx + 1, n_chunks):
        examples = load_jsonl_examples(
            filename=f'{data_dir}/chunk-{chunk_idx}.jsonl',
            n_examples=n_samples_per_chunk,
            shuffle=True,
            global_micro_batch_size=global_micro_batch_size,
            global_rank=fabric.global_rank,
            world_size=fabric.world_size)

        train_epoch(
            fabric=fabric,
            output_dir=output_dir,
            tokenizer=tokenizer,
            model=model,
            optimizer=optimizer,
            lr_schedule_fn=lr_schedule_fn,
            examples=examples,
            per_device_batch_size=per_device_batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            chunk_idx=chunk_idx,
            run_wandb=run_wandb)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    fire.Fire(main)