# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

from transformers import AutoTokenizer, AutoModel
import torch
import torch.distributions as dists
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union
    
@dataclass
class LMGenerationConfig:
    max_new_tokens: int = 128
    timesteps: int = 128

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0

    # propagate_eot_token_ids: Optional[List[int]] = None
    # propagate_eot_str: Optional[Union[str, List[str]]] = None


    def __post_init__(self):
        pass

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown generation config field: {key}")
            
    def copy(self):
        return copy.deepcopy(self)
            
            
    
@dataclass
class LLaDAConfig:
    mask_token_id: int = 126336
    endoftext_token_id: int = 126348
    
@dataclass
class DreamConfig:
    mask_token_id: int = 151666
    endoftext_token_id: int = 151643
    
@dataclass
class OpenDLMOutput:
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    NFE: Optional[int] = None     
       

class OpenDLM:
    def __init__(self, model_name, torch_dtype=torch.bfloat16, trust_remote_code=True):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)
        if "llada" in model_name.lower():
            self.dlm_config = LLaDAConfig()
        elif "dream" in model_name.lower():
            self.dlm_config = DreamConfig()
            self.model = ShiftedLogitsWrapper(self.model)
        else:
            raise ValueError(f"Model {model_name} is not supported")
        
        self.tokenizer.mask_token_id = self.dlm_config.mask_token_id
        self.tokenizer.endoftext_token_id = self.dlm_config.endoftext_token_id
        self.mask_id = self.dlm_config.mask_token_id
        self.endoftext_id = self.dlm_config.endoftext_token_id



    def generate(
        self,
        sampler,
        inputs: Optional[Union[str, torch.Tensor]] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[LMGenerationConfig] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        if inputs is not None and input_ids is None:
            enc = self.tokenizer(inputs, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))

        if input_ids is None or attention_mask is None:
            raise ValueError("Must provide `inputs` or (`input_ids` and `attention_mask`)")

        # 1. Use default if not provided
        if generation_config is None:
            generation_config = LMGenerationConfig()

        # 2. Copy and update with kwargs
        generation_config = copy.deepcopy(generation_config)
        generation_config.update(**kwargs)

        output = sampler.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=generation_config
        )

        return output if return_dict else output["sequences"]


class ShiftedLogitsWrapper(torch.nn.Module):
    """
    The ShiftedLogitsWrapper is a wrapper that shifts the logits of the model.
    It is used to shift the logits of the model to the left by one token.
    This is used to make the model output the next token in the sequence.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    @property
    def device(self):
        return next(self.model.parameters()).device

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logits = outputs.logits  # shape: [B, T, V]
        outputs.logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        return outputs


            
            
            



