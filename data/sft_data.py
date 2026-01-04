import os
import json
import glob
import datasets
from transformers import AutoTokenizer
from datasets import Features, Value, Sequence

MISTRAL_TEMPLATE = "{{ bos_token }}{% set offset = 0 %}{% if messages[0]['from'] == 'system' %}{% set offset = 1 %}{% endif %}{% for message in messages %}{% if (message['from'] == 'human') != ((loop.index0 + offset) % 2 == 0) %}{{  raise_exception('Roles in this conversation: ' + messages | map(attribute='from') | join(', ')) }}{% endif %}{% if message['from'] == 'human' %}{{ '[INST] ' + message['value'] + ' [/INST]' }}{% elif message['from'] == 'gpt' %}{{ ' ' + message['value'] + eos_token }}{% elif message['from'] == 'system' %}{{ message['value'] + '\n\n' }}{% else %}{{ raise_exception('Only system, human, and assistant roles are supported! Found: ' + message['from']) }}{% endif %}{% endfor %}"


# ---------------- 配置 ----------------
MODEL_NAME = "mistralai/Mamba-Codestral-7B-v0.1"
SHARD_SIZE = 50000
CONTEXT_LENGTH = 2048
SEED = 11111
RAW_JSONL_DIR = './crystal_chat/raw'
TOKENIZED_DIR = os.getenv("TOKENIZED_DIR", './crystal_chat/tokenized')
SHARD_DIR = os.getenv("SHARD_DIR",f'./crystal_chat/{CONTEXT_LENGTH}')

def convert_to_role_content(messages):
    mapping = {"human": "user", "gpt": "assistant", "system": "system"}
    converted = []
    for msg in messages:
        role = mapping.get(msg["from"], "user")  # 默认 user
        converted.append({"role": role, "content": msg["value"]})
    return converted

# ---------------- 处理函数 ----------------
def tokenize_chat(example, tokenizer):
    """处理 chat 格式数据"""
    # messages = convert_to_role_content(example['conversations'])
    messages = example['conversations']

    try:
        text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        )
    except Exception as e:
        print(f"Error in example: {example}")
        exit(1)
        raise e

    spans = []
    pos = 0
    for m in messages:
        if m['from'] == 'gpt':
            s = text.find("[/INST] " + m['value'], pos) + 7 # avoid match user message '[/INST]' includes 7 chars
            pos = s + len(m['value']) + 4
            spans.append((s, pos)) # +4 is for added </s>

    enc = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True)
    ids, offs = enc['input_ids'], enc['offset_mapping']

    # 再 tokenize
    mask = [1 if any(ts < e and te > s for s, e in spans) else 0 for ts, te in offs]

    # print(text)
    # print(ids)
    # print(mask)
    # idx = next((i for i, m in enumerate(mask) if m != 0), None)
    # if idx is not None:
    #     print("第一个非零位置:", idx)
    #     print("对应的id:", ids[idx])
    # else:
    #     print("没有非零元素")

    return {"token_ids": ids, "tgt_mask": mask}


def tokenize_text(example, tokenizer):
    """处理 text 格式数据"""
    ids = tokenizer(example['markdown'], add_special_tokens=True)['input_ids']
    ids = ids + [tokenizer.eos_token_id]
    return {"token_ids": ids, "tgt_mask": [1] * len(ids)}


def chunking(batch):
    """把 token_ids 拼接并切成定长 context"""
    all_ids = []
    all_mask = []
    out = {"token_ids": [], "tgt_mask": []}

    for ids, mask in zip(batch["token_ids"], batch["tgt_mask"]):
        all_ids.extend(ids)
        all_mask.extend(mask)

        while len(all_ids) >= CONTEXT_LENGTH:
            out["token_ids"].append(all_ids[:CONTEXT_LENGTH])
            out["tgt_mask"].append(all_mask[:CONTEXT_LENGTH])
            all_ids = all_ids[CONTEXT_LENGTH:]
            all_mask = all_mask[CONTEXT_LENGTH:]

    return out

def load_and_standardize_file(file_path, data_type):
    """
    分别加载单个 JSONL 文件，并统一列名为 tokenizable 列。
    对 chat 文件，保留 conversations 中每个 dict 的 'from' 和 'value' 字段。
    """
    dataset = datasets.load_dataset("json", data_files=file_path, split="train")

    if data_type == "chat":
        # 保留 columns
        keep_col = ["conversations"]
        drop_cols = [col for col in dataset.column_names if col not in keep_col]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)

        # 对 conversations list 里的每个 dict 保留 'from' 和 'value'
        def whether_clean(dataset):
            return dataset.features['conversations'][0].keys() > set(['from', 'value'])

        features = {"conversations":[{'from': Value('string'), 'value': Value('string')}]}

        def clean_conversations(example):
            example['conversations'] = [
                {'from': c['from'], 'value': c['value']} 
                for c in example['conversations']
            ]
            return example

        if whether_clean(dataset):
            print(f"Cleaning conversations in {file_path} ...")
            dataset = dataset.map(clean_conversations, num_proc=os.cpu_count(), features=Features(features))

    elif data_type == "text":
        keep_col = ["markdown"]
        drop_cols = [col for col in dataset.column_names if col not in keep_col]
        if drop_cols:
            dataset = dataset.remove_columns(drop_cols)
    else:
        raise ValueError(f"Unknown data_type {data_type}")

    return dataset


# ---------------- 主函数 ----------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.chat_template = MISTRAL_TEMPLATE

    # 1️⃣ 加载 chat 数据集
    chat_files = [f for f in glob.glob(f"{RAW_JSONL_DIR}/*.jsonl")
                if os.path.basename(f) not in ["textbooks.jsonl", "program_books.jsonl"]]
    chat_datasets = [load_and_standardize_file(f, "chat") for f in chat_files]
    
    chat_ds = datasets.concatenate_datasets(chat_datasets) if chat_datasets else None

    # 2️⃣ 加载 text 数据集
    text_files = [f for f in glob.glob(f"{RAW_JSONL_DIR}/*.jsonl") if os.path.basename(f) in ["textbooks.jsonl", "program_books.jsonl"]]
    text_datasets = [load_and_standardize_file(f, "text") for f in text_files]
    text_ds = datasets.concatenate_datasets(text_datasets) if text_datasets else None

    # 3️⃣ tokenize
    chat_ds = chat_ds.map(
        lambda x: tokenize_chat(x, tokenizer),
        remove_columns=chat_ds.column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing chat data",
        )
    text_ds = text_ds.map(
        lambda x: tokenize_text(x, tokenizer),
        remove_columns=text_ds.column_names,
        num_proc=os.cpu_count(),
        desc="Tokenizing text data",
        )

    # 4️⃣ 合并两个 dataset
    dataset = datasets.concatenate_datasets([chat_ds, text_ds])

    # 5️⃣ 拼接 & 切块
    dataset = dataset.map(
        chunking,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=dataset.column_names,
        desc=f"Chunking to {CONTEXT_LENGTH} length",
    )

    # 6️⃣ shuffle
    dataset = dataset.shuffle(seed=SEED)

    # 7️⃣ 分 shard 并保存
    os.makedirs(SHARD_DIR, exist_ok=True)
    n_shards = len(dataset) // SHARD_SIZE + 1
    for i in range(n_shards):
        shard = dataset.shard(num_shards=n_shards, index=i)
        out_file = os.path.join(SHARD_DIR, f"chunk-{i}.jsonl")
        shard.to_json(out_file, num_proc=os.cpu_count())
        print(f"✅ Saved {out_file} ({len(shard)} samples)")


if __name__ == "__main__":
    main()