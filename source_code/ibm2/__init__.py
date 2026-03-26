# -*- coding: utf-8 -*-

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from fla.models.ibm2.configuration_ibm2 import IBM2Config
from fla.models.ibm2.modeling_ibm2 import IBM2ForCausalLM, IBM2Model

AutoConfig.register(IBM2Config.model_type, IBM2Config, exist_ok=True)
AutoModel.register(IBM2Config, IBM2Model, exist_ok=True)
AutoModelForCausalLM.register(IBM2Config, IBM2ForCausalLM, exist_ok=True)


__all__ = ['IBM2Config', 'IBM2ForCausalLM', 'IBM2Model']
