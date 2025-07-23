# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .sampler import Sampler
import torch
from opendlm.model import LMGenerationConfig, OpenDLMOutput
from .sampler import Sampler
from .scheduler import UnmaskingScheduler

class BlockSampler(Sampler):
    """
    Implements the Block Sampler.
    
    Args:
        block_length: The length of the block.
        unmasking_scheduler: The unmasking scheduler.
        score_type: The score type for selecting tokens.
        propagate_eot: If set to True, all subsequent positions after the end-of-text token will be filled with the end-of-text token. 
        random_selection: If set to True, the tokens will be selected randomly rather than using the score.

    Reference:
        - Nie, Shen, et al. "Large language diffusion models." arXiv preprint arXiv:2502.09992 (2025).
    """
    def __init__(self, block_length: int = 32, unmasking_scheduler: UnmaskingScheduler = None, score_type: str = "confidence", propagate_eot: bool = False, random_selection: bool = False):
        super().__init__(unmasking_scheduler, score_type, propagate_eot, random_selection)
        self.block_length = block_length


    def generate(self, input_ids, attention_mask, model, tokenizer, generation_config: LMGenerationConfig):
        NFE = 0
        history = []
        gen_length = generation_config.max_new_tokens
        block_length = self.block_length
        timesteps = generation_config.timesteps
        mask_token_id = tokenizer.mask_token_id
        endoftext_token_id = tokenizer.endoftext_token_id
        
        if gen_length % block_length != 0:
            raise ValueError(f"gen_length {gen_length} must be divisible by block_length {block_length}")
        if timesteps % (gen_length // block_length) != 0:
            raise ValueError(f"time steps {timesteps} must be divisible by the number of blocks {gen_length // block_length}")
        
        x = torch.full((input_ids.shape[0], input_ids.shape[1] + gen_length), mask_token_id, dtype=torch.long).to(input_ids.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        

        num_blocks = gen_length // block_length
        timesteps = timesteps // num_blocks
        
        self.unmasking_scheduler.reset(block_length, timesteps)

        for num_block in range(num_blocks):
            block_start = input_ids.shape[1] + num_block * block_length
            block_end = block_start + block_length

            for i_step in range(timesteps):
                block_masked_index = (x[:, block_start:block_end] == mask_token_id)
                logits = model(x).logits
                NFE += 1
                gen_logits = logits[:, block_start:block_end]
                
                gen_block_confidence, gen_block_x = self.sample_tokens(gen_logits, generation_config.temperature, generation_config.top_p, generation_config.top_k)
                
                selected_index = self.unmasking_scheduler.get_transfer_indices(gen_block_confidence, block_masked_index, i_step)
                if selected_index.shape[0] > 0:
                    x[selected_index[:, 0], block_start + selected_index[:, 1]] = gen_block_x[selected_index[:, 0], selected_index[:, 1]]
                
                if self.propagate_eot:
                    x = self.propagate_eot_token(x, endoftext_token_id, prompt_length=input_ids.shape[1])
                history.append(x.clone())
        return OpenDLMOutput(sequences=x, history=history, NFE=NFE)
    
