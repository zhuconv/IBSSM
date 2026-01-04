import sys
from threading import Lock
import time
import os
import shutil
import glob
import json
import tqdm
import math
import numpy as np
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from lightning.fabric.strategies import FSDPStrategy


def load_jsonl_examples(filename,
                        n_examples,
                        shuffle,
                        global_micro_batch_size,
                        global_rank,
                        world_size):
    example_idxes = np.random.permutation(n_examples) if shuffle \
        else np.arange(n_examples)

    n_examples = n_examples // global_micro_batch_size * global_micro_batch_size
    example_idxes = example_idxes[global_rank:n_examples:world_size]

    examples = {idx: None for idx in example_idxes}
    for example_idx, line in tqdm.tqdm(
            enumerate(open(filename)), desc=f'loading {filename}'):
        if example_idx in examples:
            examples[example_idx] = json.loads(line)

    return [examples[idx] for idx in example_idxes]


def get_cosine_lr_decay_fn(total_steps,
                           warmup_steps,
                           learning_rate,
                           end_learning_rate):
    def cosine_with_warmup_lr(step):
        if step < warmup_steps:
            return learning_rate * step / warmup_steps
        elif step > total_steps:
            return end_learning_rate

        decay_ratio = (step - warmup_steps) / (total_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return end_learning_rate + coeff * (learning_rate - end_learning_rate)

    return cosine_with_warmup_lr


def save_checkpoint(fabric, tokenizer, model, optimizer, save_dir):
    assert isinstance(fabric.strategy, FSDPStrategy)

    save_policy = FullStateDictConfig(
        offload_to_cpu=(fabric.world_size > 1), rank0_only=True)
    with FSDP.state_dict_type(
            model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=save_policy):
        state_dict = model._forward_module.state_dict()

    if fabric.global_rank == 0:
        tokenizer.save_pretrained(save_dir)
        # assert isinstance(model.module, LlamaForCausalLM)
        model.module.save_pretrained(
            save_dir, state_dict=state_dict, safe_serialization=False)

    fabric.barrier()
    fabric.save(
        path=f'{save_dir}/fabric_ckpt',
        state={'model': model, 'optimizer': optimizer})

    if fabric.global_rank == 0:
        work_dir, current_ckpt_name = os.path.split(save_dir)
        chunk_idx = int(current_ckpt_name.split('_')[-1])
        prev_ckpt_dir = os.path.join(work_dir, f'ckpt_{chunk_idx - 1}')

        if os.path.exists(prev_ckpt_dir):
            shutil.rmtree(prev_ckpt_dir)


def get_grad_norm(model):
    square_sum = 0.
    for param in model.parameters():
        if param.grad is not None:
            square_sum += param.grad.detach().data.norm(2).item() ** 2
    return square_sum ** 0.5


def get_last_ckpt_idx(workdir):
    last_ckpt_idx = -1
    for ckpt_dir in glob.glob(f'{workdir}/ckpt_*'):
        ckpt_idx = int(ckpt_dir.split('_')[-1])
        if ckpt_idx > last_ckpt_idx:
            last_ckpt_idx = ckpt_idx

    return last_ckpt_idx

class ProgressBar(object):
    def __init__(self, total, desc='', length=40, min_update_interval=0.1):
        self.total = total
        self.length = length
        self.current = 0
        self.desc = desc
        self.lock = Lock()
        self.start_time = time.time()
        self.min_update_interval = min_update_interval
        self.last_update_time = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def update(self, advance=1):
        with self.lock:
            self.current += advance
            current_time = time.time()
            if current_time - self.last_update_time >= self.min_update_interval:
                self._render()
                self.last_update_time = current_time
            if self.current == self.total:
                self._render()
                sys.stdout.write("\n")
                sys.stdout.flush()

    def _render(self):
        elapsed_time = time.time() - self.start_time
        progress = self.current / float(self.total)
        block = int(round(self.length * progress))
        progress_percent = round(progress * 100, 2)
        color_code = "\033[91m"
        if block >= self.length:
            block = self.length
            bar = "━" * block
            color_code = "\033[92m"
        else:
            bar = "━" * block + "-" * (self.length - block)

        speed = self.current / elapsed_time if elapsed_time > 0 else 0
        remaining_time = (self.total - self.current) / speed if speed > 0 else 0
        total_time = time.time() - self.start_time

        show_extend = [
            "{:.2f}it/min".format(speed * 60),
            "ET/ETA: {:.2f}min/{:.2f}min".format(total_time / 60,remaining_time / 60),
        ]
        out_template = "\r{2} {{2:>{1}}}/{{3:<{1}}} {{0:4.1f}}% [{0}{{1}}\033[0m] {{4}}".format(color_code, len(repr(self.total)), self.desc)
        output = out_template.format(
            progress_percent, bar, self.current, self.total, ", ".join(show_extend))

        # Ensure the length of the output is consistent to avoid flickering
        sys.stdout.write(output)
        sys.stdout.flush()