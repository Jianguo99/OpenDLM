# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from .sampler import Sampler
import torch
from opendlm.model import LMGenerationConfig, OpenDLMOutput
from .scheduler import UnmaskingScheduler

class NullCFGSampler(Sampler):
    """
    Implements the Null Classifier-Free Guidance (NullCFG) Sampler.
    
    Args:
        guidance_scale: The guidance scale for the classifier-free guidance.
        unmasking_scheduler: The unmasking scheduler.
        score_type: The score type for selecting tokens.
        early_stopping: If set to True, all subsequent positions after the end-of-text token will be filled with the end-of-text token. 
        random_selection: If set to True, the tokens will be selected randomly rather than using the score.

    Reference:
        - Nie, Shen, et al. "Large language diffusion models." arXiv preprint arXiv:2502.09992 (2025).
    """
    def __init__(self, guidance_scale: float = 1.0, unmasking_scheduler: UnmaskingScheduler = None, score_type: str = "confidence", early_stopping: bool = False, random_selection: bool = False):
        super().__init__(unmasking_scheduler, score_type, early_stopping=early_stopping, random_selection=random_selection)
        self.guidance_scale = guidance_scale  # CFG weight w

    def generate(self, input_ids, attention_mask, model, tokenizer, generation_config: LMGenerationConfig):
        NFE = 0
        
        gen_length = generation_config.max_new_tokens
        timesteps = generation_config.timesteps
        mask_token_id = tokenizer.mask_token_id
        endoftext_token_id = tokenizer.endoftext_token_id
        history = []
        prompt_len = input_ids.shape[1]
        x = torch.full((input_ids.shape[0], prompt_len + gen_length), mask_token_id, dtype=torch.long).to(input_ids.device)
        x[:, :prompt_len] = input_ids.clone()
        prompt_index_full_x = (x != mask_token_id)

        for i_step in range(timesteps):
            mask_index = (x[:, prompt_len:] == mask_token_id)
            if mask_index.sum() == 0:
                break
            
            cfg_x = x.clone()
            cfg_x[prompt_index_full_x] = mask_token_id
            cfg_logits = model(cfg_x).logits
            cond_logits = model(x).logits
            guided_logits = cond_logits + self.guidance_scale * (cond_logits - cfg_logits)
            NFE += 2

            # sample from guided logits at mask positions
            gen_logits = guided_logits[:, prompt_len:]
            gen_confidence, gen_x = self.sample_tokens(gen_logits, generation_config.temperature, generation_config.top_p, generation_config.top_k)

            selected_index = self.unmasking_scheduler.get_transfer_indices(gen_confidence, mask_index, i_step)
            if selected_index.shape[0] > 0:
                x[selected_index[:, 0], prompt_len + selected_index[:, 1]] = gen_x[selected_index[:, 0], selected_index[:, 1]]
            

            if self.early_stopping:
                x = self.propagate_eot_token(x, tokenizer.endoftext_token_id, prompt_length=prompt_len)

            history.append(x.clone())

        return OpenDLMOutput(sequences=x, history=history, NFE=NFE)