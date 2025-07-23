# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from .base import UnmaskingScheduler


class LinearUnmaskingScheduler(UnmaskingScheduler):
    """
    Implements the Linear Unmasking Scheduler.
    
    Args:
        gen_length: The length of the generation sequence.
        timesteps: The number of time steps.
        
    Reference:
        - https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
    """
    def __init__(self, gen_length: int, timesteps: int):
        super().__init__()
        self.gen_length = gen_length  # L
        self.timesteps = timesteps  # R
        
        self._precompute_transfer_nums()
        
    def _precompute_transfer_nums(self):
        """precompute transfer numbers for each step"""

        base = self.gen_length // self.timesteps
        remainder = self.gen_length % self.timesteps


        self.transfer_nums =  torch.zeros(self.timesteps, dtype=torch.int64) + base
        self.transfer_nums[:remainder] += 1
            
            
    def get_transfer_nums(self, step):
        """get transfer numbers for the specified step"""
        if step >= self.timesteps:
            return torch.empty(0, dtype=torch.long)
        
        return self.transfer_nums[step]
    
    def get_transfer_indices(self, gen_confidence, masked_index, step):
        # the mask index is the index of the generation tokens
        if step >= self.timesteps:
            return torch.empty(0, dtype=torch.long, device=gen_confidence.device), torch.empty(0, dtype=torch.long, device=gen_confidence.device)
        
        masked_confidence = gen_confidence[masked_index]
        num_tokens = masked_confidence.numel()
        k = min(self.transfer_nums[step], num_tokens)
        _, select_index_in_masked = torch.topk(masked_confidence, k=k)   
        
        masked_pos = torch.nonzero(masked_index, as_tuple=False)
        selected_pos = masked_pos[select_index_in_masked]
        
        return selected_pos
    
    def reset(self, gen_length: int, timesteps: int):
        self.gen_length = gen_length
        self.timesteps = timesteps
        self._precompute_transfer_nums()
    
    
