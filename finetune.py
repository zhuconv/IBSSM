from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import fire
import torch
from torch import nn
from transformers import AutoTokenizer
from lit_gpt.model import Mamba2ForClassification, Config
import datasets
from lightning.fabric import Fabric
from lightning.fabric.loggers import TensorBoardLogger
# from finetune import get_dataset

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True
ADD_CLS_TOKEN = True
BATCH_SIZE = 16
LOSS_SCALER = 1

class GeneralDataset(Dataset):
    def __init__(
            self,
            inputs,
            targets,
            tokenizer,
            add_cls_token,
            truncation=False
    ):
        self.targets = targets
        self.tokenizer = tokenizer
        self.num_classes = len(set(self.targets))

        # 添加CLS token（使用EOS token代替）
        if add_cls_token:
            self.inputs = [inp + tokenizer.eos_token for inp in inputs]
        else:
            self.inputs = inputs
        self.tokenized_inputs = tokenizer(
            self.inputs,
            padding=True,
            return_tensors="pt",
            add_special_tokens=True
        )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        features = {
            'input_ids': self.tokenized_inputs.input_ids[idx],
            'attention_mask': self.tokenized_inputs.attention_mask[idx],
            'label': self.targets[idx]
        }
        return features

def load_model(checkpoint_path, config, num_classes, device, dtype=torch.bfloat16):
    config = Config.from_name(config)
    model = Mamba2ForClassification(config, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model

def get_dataset(
        name,
        tokenizer,
        add_cls_token=False,
        truncation=False,
        split='validation'
):
    label_col = 'label'
    if name == 'rotten_tomatoes':
        dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split=split)
        text_col = 'text'
    elif name in ['sst2', 'cola']:
        dataset = load_dataset("glue", name, split=split)
        text_col = 'sentence'
    #     dataset = dataset.map(combine, batched=True, remove_columns=['sentence1', 'sentence2'])
    #     text_col = 'text'
    # elif name in ['emotion']:
    #     dataset = load_dataset(name, split=split)
    #     text_col = 'text'
    elif name in ['imdb']:
        dataset = load_dataset("jahjinx/IMDb_movie_reviews", split=split)
        text_col = 'text'
    elif name == 'snli':
        def combine(examples):
            premise = examples['premise']
            hypothesis = examples['hypothesis']
            examples['text'] = [x + " " + y for x, y in zip(premise, hypothesis)]
            return examples
        dataset = load_dataset("snli", split=split)
        dataset = dataset.filter(lambda example: example['label'] != -1)
        dataset = dataset.map(combine, batched=True, remove_columns=['hypothesis', 'premise'])
        text_col = 'text'
    else:
        print('Using medical-bios as default dataset.')
        dataset = load_dataset("coastalcph/medical-bios", "standard", trust_remote_code=True, split=split)
        text_col = 'text'

    dataset = GeneralDataset(
        dataset[text_col],
        dataset[label_col],
        tokenizer,
        add_cls_token,
        truncation
    )
    return dataset


def validate(fabric, model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids']
            labels = batch['label']

            logits = model(input_ids).float()
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum()
            batch_size = labels.size(0)

            # 同步所有进程的指标
            gathered_loss = fabric.all_gather(loss.detach() * batch_size)
            gathered_correct = fabric.all_gather(correct)
            gathered_samples = fabric.all_gather(torch.tensor(batch_size))

            total_loss += gathered_loss.sum().item()
            total_correct += gathered_correct.sum().item()
            total_samples += gathered_samples.sum().item()

    val_loss = total_loss / total_samples
    val_acc = total_correct / total_samples
    return val_loss, val_acc

def load_model(checkpoint_path, config, num_classes, fabric):
    config = Config.from_name(config)
    model = Mamba2ForClassification(config, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"], strict=False)
    return model


def main(config_name, dataset_name, num_epochs=3, add_cls_token=True):
    # 初始化Fabric
    logger = TensorBoardLogger(root_dir=f"logs/{dataset_name}")
    fabric = Fabric(
        accelerator="auto",
        devices="auto",
        strategy="auto",
        precision="bf16-true",
        loggers=logger
    )
    fabric.launch()
    fabric.seed_everything(42 + fabric.global_rank)

    # 只在主进程打印信息
    if fabric.is_global_zero:
        print(f"Training on {fabric.world_size} device(s)")

    # 加载数据
    tokenizer = AutoTokenizer.from_pretrained("Orkhan/llama-2-7b-absa", trust_remote_code=True)
    train_dataset = get_dataset(dataset_name, tokenizer, add_cls_token=add_cls_token, split='train')
    val_dataset = get_dataset(dataset_name, tokenizer, add_cls_token=add_cls_token, split='validation')

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = load_model(
        f"../../pretrain_checkpoints/tsz512x4k_20B_{config_name}/model.pth",
        config_name,
        train_dataset.num_classes,
        fabric
    )
    if add_cls_token:
        from functools import partial
        model.forward = partial(model.forward, add_cls_token=True)

    # 参数分组
    cls_head_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "cls_head" in name:
            cls_head_params.append(param)
        else:
            other_params.append(param)

    # 优化器设置
    optimizer = torch.optim.AdamW([
        {'params': cls_head_params, 'lr': 4e-4},
        {'params': other_params, 'lr': 4e-5}
    ])
    criterion = nn.CrossEntropyLoss()

    # Fabric包装
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids']
            labels = batch['label']

            # 前向传播
            logits = model(input_ids)
            ce_loss = criterion(logits.float(), labels)

            # 辅助损失
            auxiliary_loss = []
            def get_auxiliary_loss(module):
                if hasattr(module, "get_auxiliary_loss"):
                    auxiliary_loss.append(module.get_auxiliary_loss())
            model.apply(get_auxiliary_loss)
            kl_loss = sum(auxiliary_loss) * LOSS_SCALER
            loss = ce_loss + kl_loss

            # 反向传播
            fabric.backward(loss)
            optimizer.step()

            # 指标计算
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum()
            batch_size = labels.size(0)

            # 同步指标
            gathered_loss = fabric.all_gather(loss.detach() * batch_size)
            gathered_correct = fabric.all_gather(correct)
            gathered_samples = fabric.all_gather(torch.tensor(batch_size))

            total_loss += gathered_loss.sum().item()
            total_correct += gathered_correct.sum().item()
            total_samples += gathered_samples.sum().item()

            # 日志记录
            if batch_idx % 10 == 0 and fabric.is_global_zero:
                batch_loss = loss.item()
                batch_acc = (correct / batch_size).item()
                fabric.print(
                    f"Epoch [{epoch+1}/{num_epochs}] | "
                    f"Batch [{batch_idx}/{len(train_loader)}] | "
                    f"Loss: {batch_loss:.4f} | "
                    f"Acc: {batch_acc:.4f} | "
                    f"KL Loss: {kl_loss.item():.4f}"
                )

        # 验证和保存
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        val_loss, val_acc = validate(fabric, model, val_loader, criterion)

        fabric.print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        fabric.print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        fabric.print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        fabric.print("=================================================")

    # 保存检查点
    path = 'finetune_checkpoints'
    # path = "../../finetune_checkpoints"
    fabric.save(
        f"{path}/{config_name}_{dataset_name}_epoch{num_epochs}.pth",
        {"model": model}
    )

if __name__ == '__main__':
    fire.Fire(main)