from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.hybrid_ibm2.configuration_hybrid_ibm2 import HybridIBM2Config
from fla.models.hybrid_ibm2.modeling_hybrid_ibm2 import HybridIBM2ForCausalLM, HybridIBM2Model

AutoConfig.register(HybridIBM2Config.model_type, HybridIBM2Config, exist_ok=True)
AutoModel.register(HybridIBM2Config, HybridIBM2Model, exist_ok=True)
AutoModelForCausalLM.register(HybridIBM2Config, HybridIBM2ForCausalLM, exist_ok=True)

__all__ = ['HybridIBM2Config', 'HybridIBM2ForCausalLM', 'HybridIBM2Model']
