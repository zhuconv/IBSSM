from argparse import Namespace
from transformers import AutoTokenizer
import os
from pretrain import get_processed_dataset, get_streaming_dataset

args = Namespace()
ctl = args.context_len = 4096
CPU_COUNT = os.cpu_count()
dpt = args.dataset_cache_dir = "../../hf_datasets/SlimPajama-627B"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)

lm_datasets = get_streaming_dataset(tokenizer, args, args, cached='grouped')

print(lm_datasets)