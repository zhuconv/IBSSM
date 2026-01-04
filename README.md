### Environment

```bash
pip install -r requirements.txt

git clone https://github.com/fla-org/flash-linear-attention.git
cd flash-linear-attention
git checkout v0.1.2
pip install -e .
ln -s "$(pwd)/ibm2_source" fla/models/ibm2
```

### Data Preparation
We use [cerebras/SlimPajama-627B](https://huggingface.co/datasets/cerebras/SlimPajama-627B) to pretrain ~400M models and [CrystalChat](https://drive.google.com/file/d/1tJyBuBNus0KDMEI0CvCrpLpEeUX1c6FI/view) to sft ~7B models. SlimPajama should be manually downloaded into any path and specified via flag ``--dataset_cache_dir``. CrystalChat should be manually downloaded into ``./data`` and then preprocessed via ``./data/sft_data.py``.


### Training
Pretraining uses
```bash
bash script/pretrain.sh
```

SFT uses
```bash
bash script/posttrain.sh
```

### Evaluation
Pretraining Evaluation Harness uses
```bash
bash script harness.sh # set method_list as names of methods
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