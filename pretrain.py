#    Modification Copyright 2024 Jiajun Zhu
#    Modification Copyright 2024 Zhenyu He
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import builtins
import logging
import os
import math
import glob
import random
from itertools import chain
from dataclasses import dataclass, field
from typing import Optional

# import wandb
import transformers
from transformers import Trainer, default_data_collator, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset, load_from_disk, IterableDataset

from fla.models import *
from fla.models.ibm2 import *
from fla.models.hybrid_gated_deltanet import *
from fla.models.hybrid_mamba2 import *
from fla.models.hybrid_ibm2 import *

FLA_MODEL_NAME_MAPPING = {
    'rwkv6': 'RWKV6',
    'gla': 'GLA',
    'retnet': 'RetNet',
    'mamba': 'Mamba',
    'gated_deltanet': 'GatedDeltaNet',
    'mamba2': 'Mamba2',
    'samba': 'Samba',
}

HYBRID_MODEL_NAME_MAPPING = {
    'hybrid_gated_deltanet': 'HybridGatedDeltaNet',
    'hybrid_mamba2': 'HybridMamba2',
}

IB_MODEL_NAME_MAPPING = {
    'ib2': 'IBM2',
    'bibs2': 'BIBS2',
    'hybrid_ibm2': 'HybridIBM2',
}

CPU_COUNT = os.cpu_count()


@dataclass
class ModelArguments:
    config_name: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    dataset_cache_dir: str = field(default=None, metadata={"help": "Path to the data."})
    dataset_cached: str = field(default="grouped", metadata={"help": "Dataset cache mode: raw, tokenized, grouped, or huggingface."})
    dataset_subset: Optional[str] = field(default=None, metadata={"help": "Subset/config name for HuggingFace datasets (e.g. sample-10BT)."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    context_len: int = field(
        default=2048,
        metadata={"help": "Training Context Length."},
    )
    resume_from_checkpoint: Optional[bool] = field(default=None)
    finetune_from_pretrained: Optional[str] = field(default=None)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def get_processed_dataset(tokenizer, data_args, training_args, cached='tokenized'):

    # "../../hf_datasets/SlimPajama-627B"
    dpt = data_args.dataset_cache_dir

    assert cached in ['raw', 'tokenized', 'grouped'], "cached should be one of ['raw', 'tokenized', 'grouped']"
    if cached == 'grouped':
        print("Loading datasets: Grouped")
        lm_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_test_*.arrow"},
            num_proc=CPU_COUNT,
            split=None
        )
        return lm_datasets

    elif cached == 'raw':
        raw_datasets = load_dataset("json",  # 本地路径
            data_files={
                "train": f"{dpt}/train/*/*.jsonl.zst",
                "validation": f"{dpt}/validation/*/*.jsonl.zst",
                "test": f"{dpt}/test/*/*.jsonl.zst"
            },
            num_proc=CPU_COUNT,
            split=None,
        )

        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])


        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
            num_proc=CPU_COUNT,
            load_from_cache_file=True,
            cache_file_names={"train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train.arrow",\
                "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation.arrow", \
                "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test.arrow"},
            desc="Running tokenizer on dataset",
        )
    elif cached == 'tokenized':
        tokenized_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test_*.arrow"
            },
            num_proc=CPU_COUNT,
            split=None
        )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // training_args.context_len) * training_args.context_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + training_args.context_len] for i in range(0, total_length, training_args.context_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(f"{data_args.dataset_cache_dir}/{training_args.context_len}", exist_ok=True)

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=CPU_COUNT,
        load_from_cache_file=True,
        cache_file_names={"train": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_train.arrow",\
            "validation": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_validation.arrow", \
            "test": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_test.arrow"},
        desc=f"Grouping texts in chunks of {training_args.context_len}",
    )
    return lm_datasets

def get_streaming_dataset(tokenizer, data_args, training_args, cached='tokenized'):

    # "../../hf_datasets/SlimPajama-627B"
    dpt = data_args.dataset_cache_dir

    assert cached in ['raw', 'tokenized', 'grouped', 'huggingface'], "cached should be one of ['raw', 'tokenized', 'grouped', 'huggingface']"

    if cached == 'huggingface':
        # Load directly from a HuggingFace dataset hub name (e.g. "HuggingFaceFW/fineweb-edu")
        subset = data_args.dataset_subset
        raw_datasets = load_dataset(dpt, subset, streaming=True) if subset else load_dataset(dpt, streaming=True)

        column_names = list(next(iter(raw_datasets["train"])).keys())
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // training_args.context_len) * training_args.context_len
            result = {
                k: [t[i : i + training_args.context_len] for i in range(0, total_length, training_args.context_len)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = {}
        available_splits = list(raw_datasets.keys()) if hasattr(raw_datasets, 'keys') else ["train"]
        if "validation" not in available_splits:
            logging.warning(f"No validation split found in {dpt}. Eval/predict will be disabled.")
        for split_name in available_splits:
            tokenized = raw_datasets[split_name].map(tokenize_function, batched=True, remove_columns=column_names)
            grouped = tokenized.map(group_texts, batched=True)
            lm_datasets[split_name] = grouped

        return lm_datasets

    elif cached == 'grouped':
        print("Loading datasets: Grouped")
        lm_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/{training_args.context_len}/lm_datasets_test_*.arrow"},
            split=None,
            streaming=True
        )
        return lm_datasets

    elif cached == 'tokenized':
        tokenized_datasets = load_dataset(
            "arrow",
            data_files={
                "train": f"{data_args.dataset_cache_dir}/tokenized_datasets_train_*.arrow",
                "validation": f"{data_args.dataset_cache_dir}/tokenized_datasets_validation_*.arrow",
                "test": f"{data_args.dataset_cache_dir}/tokenized_datasets_test_*.arrow"
            },
            split=None,
            streaming=True
        )

    else: # raw
        raw_datasets = load_dataset("json",  # 本地路径
            data_files={
                "train": f"{dpt}/train/*/*.jsonl.zst",
                "validation": f"{dpt}/validation/*/*.jsonl.zst",
                "test": f"{dpt}/test/*/*.jsonl.zst"
            },
            # num_proc=CPU_COUNT,
            split=None,
            streaming=True
        )

        def infer_columns_of_dataset(raw_datasets):
            default_cols = raw_datasets.features
        
            if default_cols is not None:
                return list(default_cols)
        
            first_example = next(iter(raw_datasets))
            if isinstance(first_example, dict):
                return list(first_example.keys())
            else:
                raise ValueError(f'Unable to infer column names from the data type: {type(first_example)}')


        # column_names = raw_datasets["train"].column_names
        column_names = infer_columns_of_dataset(raw_datasets["train"])
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])
        

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=column_names,
        )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // training_args.context_len) * training_args.context_len
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + training_args.context_len] for i in range(0, total_length, training_args.context_len)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(f"{data_args.dataset_cache_dir}/{training_args.context_len}", exist_ok=True)

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
    )

    return lm_datasets

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    def make_rank0_print(training_args):
        def _print(*args, **kwargs):
            if training_args.process_index == 0:
                builtins.print(*args, **kwargs)
        return _print

    #! affecting all processes
    print = make_rank0_print(training_args)

    #! Config and Model
    count_func = lambda model: sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    if model_args.config_name:
        actual_name = model_args.config_name
        ib_type = None
        if actual_name in ['ibm2b', 'ibm2g']:
            ib_type = 'bernoulli' if actual_name == 'ibm2b' else 'gamma'
            actual_name = 'ibm2'
        elif actual_name in ['hybrid_ibm2b', 'hybrid_ibm2g']:
            ib_type = 'bernoulli' if actual_name == 'hybrid_ibm2b' else 'gamma'
            actual_name = 'hybrid_ibm2'

        config = AutoConfig.for_model(actual_name, hidden_size=1024)
        if actual_name in ['mamba2', 'ibm2', 'hybrid_mamba2', 'hybrid_ibm2']:
            config.num_heads = 32
        if ib_type is not None:
            config.ib_type = ib_type
            config.auxiliary_loss_weight = 1e-6 if ib_type == 'gamma' else 1e-1
        model = AutoModelForCausalLM.from_config(config)
        # if training_args.local_rank == 0:
        print(f"Training new model from scratch - Total Size={count_func(model)/2**20:.2f}M parameters")
    elif model_args.model_name_or_path:
        # config = AutoConfig.from_pretrained(model_args.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path)
        # if training_args.local_rank == 0:
        print(f"Finetuning model from {model_args.model_name_or_path} - Model Size={count_func(model)/2**20:.2f}M parameters")
    else:
        raise NotImplementedError

    # determine if load from pretrained
    # if training_args.finetune_from_pretrained:
    #     pretrained_model = LlamaForCausalLM.from_pretrained(training_args.finetune_from_pretrained)
    #     checkpoint = pretrained_model.state_dict()
    #     def filter(key):
    #         rotary = 'sin_cached' not in key and 'cos_cached' not in key
    #         post_linear = "post_attention_linears" not in key
    #         pe_proj = "pe.proj" not in key
    #         return all((rotary, post_linear, pe_proj))
    #     filtered_checkpoint = {k: v for k, v in checkpoint.items() if filter(k)}
    #     model.load_state_dict(filtered_checkpoint, strict=False)

    # tokenizer = AutoTokenizer.from_pretrained(
    #     "/cusp-data-efa/peihaow/hf_models/llama-tokenizer",
    #     use_fast=True,
    # )
 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)

    # if training_args.local_rank > 0: 
    #     torch.distributed.barrier()

    lm_datasets = get_streaming_dataset(tokenizer, data_args, training_args, cached=data_args.dataset_cached)

    print(f"*** Datasets Loaded ***")

    train_dataset = lm_datasets["train"]
    valid_dataset = lm_datasets.get("validation", None)

    if valid_dataset is None:
        print("WARNING: No validation split available. Setting do_eval=False and do_predict=False.")
        training_args.do_eval = False
        training_args.do_predict = False

    # if training_args.local_rank == 0:
    #     torch.distributed.barrier()

    data_collator = default_data_collator # DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    data_module = dict(
        train_dataset=train_dataset, eval_dataset=valid_dataset, data_collator=data_collator
        )

    # Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    #! For Iteratable: do not skip streaming dataset but use a new shuffle for resume.
    n_lastest_iter = 0
    if training_args.resume_from_checkpoint == True:
        # search for the latest checkpoint
        from pathlib import Path
        all_checkpoints = list(Path(training_args.output_dir).glob("checkpoint-*"))
        all_checkpoints = [x for x in all_checkpoints if (x / "trainer_state.json").exists() and not x.name.endswith("final")]
        if len(all_checkpoints) == 0:
            training_args.resume_from_checkpoint = None
            print("No checkpoint found, starting from scratch")
        else:
            all_checkpoints = [str(x) for x in all_checkpoints]
            latest_checkpoint = max(all_checkpoints, key=os.path.getctime)
            training_args.resume_from_checkpoint = latest_checkpoint
            print("Resuming from checkpoint", latest_checkpoint)
            n_lastest_iter = int(latest_checkpoint.split('-')[-1])

    if isinstance(train_dataset, IterableDataset):
        shuffle_seed = training_args.data_seed + n_lastest_iter if training_args.data_seed is not None else training_args.seed + n_lastest_iter
        train_dataset = train_dataset.shuffle(seed=shuffle_seed)
        training_args.ignore_data_skip = True
        print("*** Set ignore_data_skip=True for streaming mode to save time ***")



    trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    if training_args.do_train:
        logging.info("*** Start Training ***")
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_state()
        # trainer.save_model(output_dir=training_args.output_dir)
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    if training_args.do_eval:
        logging.info("*** Evaluate on valid set***")
        metrics = trainer.evaluate(eval_dataset=valid_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        

if __name__ == "__main__":
    # wandb.init(
    #     project="IBSSM",
    #     entity="jiajun_vita",
    #     id=os.getenv("SLURM_JOB_NAME", "interact"),
    #     resume='allow',
    #     )
    transformers.logging.set_verbosity_warning()
    train()