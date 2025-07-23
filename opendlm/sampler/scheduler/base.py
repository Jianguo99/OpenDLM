# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import torch

class UnmaskingScheduler(ABC):
    """
    Abstract base class for all unmask schedulers.
    """
    
    def __init__(self):
        pass
    
    @abstractmethod
    def get_transfer_indices(
        self,
        gen_confidence: torch.Tensor,
        masked_index: torch.Tensor,
        step: int
    ) -> torch.Tensor:
        """
        Determine the token indices to be transferred at the given decoding step.

        Args:
            gen_confidence (torch.Tensor): Confidence scores for generation tokens,
                with shape [batch_size, gen_length].
            masked_index (torch.Tensor): Boolean mask indicating positions of masked tokens
                within the generation sequence, shape [batch_size, gen_length].
            step (int): Current decoding step (0-based index).

        Returns:
            torch.Tensor: Indices of the tokens selected for transfer in the
                generation sequence, shape [batch_size, num_selected].
        """
        raise NotImplementedError
        
    
