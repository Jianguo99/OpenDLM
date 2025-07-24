# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from .base import UnmaskingScheduler

class DetailedUnmaskingScheduler(UnmaskingScheduler):
    """
    Implements the Detailed Unmasking Scheduler.
    
    Args:
        gen_length: The length of the generation sequence.
        alpha: The alpha parameter for the Detailed Unmasking Scheduler.
        threshold: The threshold for the Detailed Unmasking Scheduler.
        
    Reference:
        - Luxembourg, Omer, Haim Permuter, and Eliya Nachmani. "Plan for Speed--Dilated Scheduling for Masked Diffusion Language Models." arXiv preprint arXiv:2506.19037 (2025).
    """
    def __init__(self, gen_length: int, alpha: float = 2.0, threshold: float = 0.0):
        super().__init__()
        self.gen_length = gen_length  # B
        self.alpha = alpha  # a
        self.threshold = threshold  # skip threshold
        
        # total time steps: R = ⌈log_a B⌉
        self.time_steps = math.ceil(math.log(gen_length) / math.log(alpha))
        
        # precompute transfer indices for each step
        self.transfer_indices = []
        self._precompute_transfer_indices()
        
    def reset(self, gen_length: int, timesteps: int):
        self.gen_length = gen_length
        self.time_steps = math.ceil(math.log(gen_length) / math.log(self.alpha))
        self.transfer_indices = []
        self._precompute_transfer_indices()
        
    def _precompute_transfer_indices(self):
        """precompute transfer indices for each step"""
        U_t = set()  # cumulative unmasked group
        
        for t in range(1, self.time_steps + 1):
            # step size: s_t = ⌊B/a^t⌋
            s_t = max(1, int(self.gen_length // (self.alpha ** t)))
            
            # P_t = {k ∈ {1,...,B} \ U_{t-1} | (k-1) mod s_t = 0}
            P_t = []
            for k in range(1, self.gen_length + 1):
                if k not in U_t and (k - 1) % s_t == 0:
                    P_t.append(k - 1)  # convert to 0-indexed
            
            # if the last step and P_R is smaller, merge it to the previous step to balance the coverage
            if t == self.time_steps and len(self.transfer_indices) > 0:
                if len(P_t) < len(self.transfer_indices[-1]):
                    # merge P_R to P_{R-1}
                    self.transfer_indices[-1].extend(P_t)
                    continue
            
            self.transfer_indices.append(P_t)
            
            # update cumulative unmasked group: U_t = U_{t-1} ∪ P_t
            U_t.update([k + 1 for k in P_t])  # convert to 1-indexed for tracking
    
    def get_transfer_indices(self, gen_confidence: torch.Tensor, masked_index: torch.Tensor, step: int) -> torch.Tensor:
        """
        Returns selected indices for current step as [N, 2] pairs (row, col),
        applying skip mechanism based on confidence threshold.
        """
        if step >= len(self.transfer_indices):
            return torch.empty(0, dtype=torch.long, device=gen_confidence.device)
        
        # Candidate column indices for this step
        col_idx = torch.tensor(self.transfer_indices[step], dtype=torch.long, device=gen_confidence.device)
        num_cols = col_idx.shape[0]
        batch_size = gen_confidence.shape[0]

        # Compute confidence for selected tokens
        # gen_confidence shape: [batch, seq_len]
        conf_scores = gen_confidence[:, col_idx]  # shape: [batch, num_cols]

        # Apply skip mechanism if threshold > 0
        if self.threshold > 0:
            keep_mask = (conf_scores >= self.threshold)
        else:
            keep_mask = torch.ones_like(conf_scores, dtype=torch.bool)

        # Prepare row/col pairs for kept tokens
        kept_pairs = []

        for b in range(batch_size):
            kept_cols = col_idx[keep_mask[b]]
            if len(kept_cols) > 0:
                rows = torch.full((len(kept_cols),), b, device=gen_confidence.device, dtype=torch.long)
                kept_pairs.append(torch.stack([rows, kept_cols], dim=1))
            
            # Collect skipped tokens for next step
            skip_cols = col_idx[~keep_mask[b]]
            if len(skip_cols) > 0 and step + 1 < len(self.transfer_indices):
                self.transfer_indices[step + 1].extend(skip_cols.tolist())

        if len(kept_pairs) == 0:
            return torch.empty(0, dtype=torch.long, device=gen_confidence.device)
        
        
        return torch.cat(kept_pairs, dim=0)
    