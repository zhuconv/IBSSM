from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.hybrid_gated_deltanet.configuration_hybrid_gated_deltanet import HybridGatedDeltaNetConfig
from fla.models.hybrid_gated_deltanet.modeling_hybrid_gated_deltanet import HybridGatedDeltaNetForCausalLM, HybridGatedDeltaNetModel

AutoConfig.register(HybridGatedDeltaNetConfig.model_type, HybridGatedDeltaNetConfig, exist_ok=True)
AutoModel.register(HybridGatedDeltaNetConfig, HybridGatedDeltaNetModel, exist_ok=True)
AutoModelForCausalLM.register(HybridGatedDeltaNetConfig, HybridGatedDeltaNetForCausalLM, exist_ok=True)

__all__ = ['HybridGatedDeltaNetConfig', 'HybridGatedDeltaNetForCausalLM', 'HybridGatedDeltaNetModel']
