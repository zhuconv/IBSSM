from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.hybrid_mamba2.configuration_hybrid_mamba2 import HybridMamba2Config
from fla.models.hybrid_mamba2.modeling_hybrid_mamba2 import HybridMamba2ForCausalLM, HybridMamba2Model

AutoConfig.register(HybridMamba2Config.model_type, HybridMamba2Config, exist_ok=True)
AutoModel.register(HybridMamba2Config, HybridMamba2Model, exist_ok=True)
AutoModelForCausalLM.register(HybridMamba2Config, HybridMamba2ForCausalLM, exist_ok=True)

__all__ = ['HybridMamba2Config', 'HybridMamba2ForCausalLM', 'HybridMamba2Model']
