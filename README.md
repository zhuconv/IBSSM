### Environment

```bash
pip install -r requirements.txt

git clone https://github.com/fla-org/flash-linear-attention.git
cd flash-linear-attention
git checkout v0.1.2
pip install -e .
```

#### Symlink custom models into FLA

```bash
FLA_MODELS=$(python -c "import fla, os; print(os.path.join(os.path.dirname(fla.__file__), 'models'))")
ln -sf $(pwd)/source_code/ibm2 $FLA_MODELS/ibm2
ln -sf $(pwd)/source_code/hybrid_gated_deltanet $FLA_MODELS/hybrid_gated_deltanet
ln -sf $(pwd)/source_code/hybrid_mamba2 $FLA_MODELS/hybrid_mamba2
ln -sf $(pwd)/source_code/hybrid_ibm2 $FLA_MODELS/hybrid_ibm2
```

### Available Models

| Model | Type | Config Name | Description |
|-------|------|-------------|-------------|
| GLA | Pure | `gla` | Gated Linear Attention |
| RWKV6 | Pure | `rwkv6` | RWKV v6 |
| RetNet | Pure | `retnet` | Retentive Network |
| Mamba | Pure | `mamba` | Mamba (v1) |
| Mamba2 | Pure | `mamba2` | Mamba v2 |
| GatedDeltaNet | Pure | `gated_deltanet` | Gated Delta Network |
| Samba | Hybrid | `samba` | Mamba + SWA (FLA built-in) |
| IBM2 | IB | `ibm2b` / `ibm2g` | Mamba2 + Information Bottleneck (Bernoulli/Gamma) |
| HybridGatedDeltaNet | Hybrid | `hybrid_gated_deltanet` | GatedDeltaNet + Sliding Window Attention |
| HybridMamba2 | Hybrid | `hybrid_mamba2` | Mamba2 + Sliding Window Attention |
| HybridIBM2 | Hybrid+IB | `hybrid_ibm2b` / `hybrid_ibm2g` | IBM2 + SWA (Bernoulli/Gamma IB) |

Hybrid models interleave recurrence layers with Sliding Window Attention (SWA) at configurable layer indices (default: odd layers), following the SamBa pattern.

### Data Preparation
We use [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (sample-10BT subset, streamed) to pretrain ~400M models and [CrystalChat](https://drive.google.com/file/d/1tJyBuBNus0KDMEI0CvCrpLpEeUX1c6FI/view) to sft ~7B models. The dataset is loaded via streaming by default. CrystalChat should be manually downloaded into ``./data`` and then preprocessed via ``./data/sft_data.py``.


### Training
Pretraining uses
```bash
# Pure models
METHOD=gla bash script/pretrain.sh
METHOD=mamba2 bash script/pretrain.sh

# Hybrid models
METHOD=hybrid_gated_deltanet bash script/pretrain.sh
METHOD=hybrid_mamba2 bash script/pretrain.sh
METHOD=hybrid_ibm2g bash script/pretrain.sh
```

SFT uses
```bash
bash script/posttrain.sh
```

### Evaluation
Pretraining Evaluation Harness uses
```bash
bash script/harness.sh # set method_list as names of methods
```
Pretraining Evaluation Fidelity uses
```bash
python utils/fidelity_ft.py
python eval_fidelity.py
```


SFT Evaluation Harness uses
```bash
bash script/harness.sh # set method_list as checkpoint paths of sft models
```
SFT Evaluation Robustness uses
```bash
bash script/robustness.sh
```
