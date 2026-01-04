
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pretrain import FLA_MODEL_NAME_MAPPING

from fla.models import GLAConfig, RWKV6Config, RetNetConfig, MambaConfig, GatedDeltaNetConfig
for method, model in FLA_MODEL_NAME_MAPPING.items():

# hf_model_list = ['RWKV6', 'GLA', 'RetNet', 'Mamba', "GatedDeltaNet"]
# model = 'GatedDeltaNet'
    checkpoint_path = f"../IBMamba/out/tsz512x4k_20B_{model}/model.pth"
    # GatedDeltaNetConfig = partial(GatedDeltaNetConfig, max_position_embeddings=4096)
    config = eval(f"{model}Config")(hidden_size=1024)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])

    print(f"Saving model: {method}")
    model.save_pretrained(f"./output/{method}", safe_serialization=True)

    tokenizer_name = "../../hf_models/llama-tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    tokenizer.save_pretrained(f"./output/{method}")
