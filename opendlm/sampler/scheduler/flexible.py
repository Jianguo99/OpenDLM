# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from .base import UnmaskingScheduler


class FlexibleUnmaskingScheduler(UnmaskingScheduler):
    """
    Unmasking scheduler supporting multiple scheduling strategies:
    ['linear', 'cosine', 'square', 'cubic', 'exponential', 'sqrt', 'log'].

    Args:
        gen_length (int): Total number of tokens (L)
        timesteps (int): Number of decoding steps (T)
        schedule_type (str): Scheduling type
        exp_k (float): Parameter for exponential schedule
        log_a (float): Parameter for logarithmic schedule
        
    Reference:
        - Chang, Huiwen, et al. "Maskgit: Masked generative image transformer." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
    """

    def __init__(self, gen_length: int, timesteps: int, schedule_type: str = 'linear',
                 exp_k: float = 5, log_a: float = 9):
        super().__init__()
        self.gen_length = gen_length
        self.timesteps = timesteps
        self.schedule_type = schedule_type
        self.exp_k = exp_k
        self.log_a = log_a

        self._precompute_transfer_nums()

    def _gamma(self, r: float) -> float:
        """Compute mask ratio γ(r) for given normalized progress r ∈ [0,1]."""
        if self.schedule_type == 'linear':
            return 1 - r
        elif self.schedule_type == 'cosine':
            return np.cos((np.pi / 2) * r)
        elif self.schedule_type == 'square':
            return (1 - r) ** 2
        elif self.schedule_type == 'cubic':
            return (1 - r) ** 3
        elif self.schedule_type == 'exponential':
            val = np.exp(-self.exp_k * r)
            return (val - np.exp(-self.exp_k)) / (1 - np.exp(-self.exp_k))  # Normalize to [0,1]
        elif self.schedule_type == 'sqrt':
            return np.sqrt(1 - r)
        elif self.schedule_type == 'log':
            return 1 - (np.log(1 + self.log_a * r) / np.log(1 + self.log_a))
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _precompute_transfer_nums(self):
        """
        Precompute number of tokens to unmask at each step based on schedule.
        Ensure that transfer_nums[i] = number of tokens newly revealed at step i.
        """
        # Compute mask ratio for steps [0..T]
        ratios = [self._gamma(t / self.timesteps) for t in range(self.timesteps + 1)]
        ratios[-1] = 0.0  # Ensure last step = 0
        total_tokens = self.gen_length

        # Convert ratios to absolute masked token counts
        mask_counts = [int(round(r * total_tokens)) for r in ratios]

        # Δ_t = mask_{t-1} - mask_t
        transfer_nums = []
        for t in range(self.timesteps):
            delta = max(0, mask_counts[t] - mask_counts[t + 1])
            transfer_nums.append(delta)

        self.transfer_nums = torch.tensor(transfer_nums, dtype=torch.int64)
        self.mask_ratios = ratios  # Save for visualization

    def get_transfer_nums(self, step: int):
        """Return number of tokens to unmask at given step."""
        if step >= self.timesteps:
            return torch.tensor(0, dtype=torch.int64)
        return self.transfer_nums[step]

    def get_transfer_indices(self, gen_confidence, masked_index, step):
        """Select top-k indices to unmask based on confidence."""
        if step >= self.timesteps:
            return torch.empty(0, dtype=torch.long, device=gen_confidence.device)

        masked_confidence = gen_confidence[masked_index]
        num_tokens = masked_confidence.numel()
        k = min(self.transfer_nums[step].item(), num_tokens)

        if k == 0:
            return torch.empty(0, dtype=torch.long, device=gen_confidence.device)

        _, select_index_in_masked = torch.topk(masked_confidence, k=k)
        masked_pos = torch.nonzero(masked_index, as_tuple=False)
        selected_pos = masked_pos[select_index_in_masked]
        return selected_pos

    def reset(self, gen_length: int, timesteps: int):
        """Reset scheduler with new parameters."""
        self.gen_length = gen_length
        self.timesteps = timesteps
        self._precompute_transfer_nums()
