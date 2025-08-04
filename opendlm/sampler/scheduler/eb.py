# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .base import UnmaskingScheduler

class EntropyBoundedUnmaskingScheduler(UnmaskingScheduler):
    """
    Entropy-Bounded Unmasking Scheduler (EB-Unmasking)
    

    Key idea:
        Dynamically determine the number of tokens to unmask at each step by 
        bounding joint dependence error using an entropy-based criterion.

    Args:
        gamma (float): Entropy bound threshold controlling how many tokens to unmask each step.
                       Smaller gamma -> fewer tokens (safer but slower),
                       Larger gamma -> more tokens (faster but riskier).
                       
    Reference:
        Heli Ben-Hamu, Itai Gat, Daniel Severo, Niklas Nolte, Brian Karrer (FAIR at Meta), 2025. https://arxiv.org/abs/2505.24857
    """

    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def get_transfer_indices(self, gen_entropy, gen_confidence, masked_index, step):
        """
        Select tokens to unmask based on:
        1. score_type: sorting strategy (entropy, confidence, margin)
        2. EB condition: sum(H) - max(H) <= gamma
        
        Args:
            gen_entropy (Tensor): Entropy for each token (shape: [seq_len])
            gen_confidence (Tensor): Proxy score (confidence/margin/entropy) for sorting (shape: [seq_len])
            masked_index (Tensor): Boolean mask of positions still masked
            step (int): Current step index (unused)
        
        Returns:
            Tensor: Positions to unmask (shape: [k, 1])
        """
        device = gen_entropy.device

        # Get masked positions
        masked_positions = torch.nonzero(masked_index, as_tuple=False)
        if masked_positions.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        masked_entropy = gen_entropy[masked_index]
        masked_confidence = gen_confidence[masked_index]


        sorted_confidence, sort_idx = torch.sort(masked_confidence, dim=0, descending=True)
        sorted_positions = masked_positions[sort_idx]
        sorted_entropy = masked_entropy[sort_idx]  # EB condition always uses entropy order

        # Apply EB condition: sum(H) - max(H) <= gamma
        acc_entropy = torch.cumsum(sorted_entropy, dim=0)
        cummax_entropy = torch.cummax(sorted_entropy, dim=0).values
        condition = (acc_entropy - cummax_entropy) <= self.gamma

        k = condition.sum().item()
        if k == 0:
            k = 1  # At least unmask one token per step

        return sorted_positions[:k]
    
    def reset(self, block_length: int, timesteps: int):
        """Reset scheduler with new parameters."""
        self.block_length = block_length
        self.timesteps = timesteps
        self.gamma = self.gamma


