# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from .sampler import Sampler
import torch
from opendlm.model import LMGenerationConfig, OpenDLMOutput
from .scheduler import UnmaskingScheduler

class AdaptiveCFGSampler(Sampler):
    """
    Implements the Adaptive Classifier-Free Guidance (AdaptiveCFG) Sampler.
    
    Args:
        guidance_scale: The guidance scale for the classifier-free guidance.
        remask_ratio: The ratio of the tokens to be re-masked in each step.
        unmasking_scheduler: The unmasking scheduler.
        score_type: The score type for selecting tokens.
        early_stopping: If set to True, all subsequent positions after the end-of-text token will be filled with the end-of-text token. 
        random_selection: If set to True, the tokens will be selected randomly rather than using the score.

    Reference:
        Li P, Yan S, Tsai J, et al. Adaptive Classifier-Free Guidance via Dynamic Low-Confidence Masking[J]. arXiv preprint arXiv:2505.20199, 2025
    """
    def __init__(self, guidance_scale: float = 1.0, remask_ratio: float = 0.3, unmasking_scheduler: UnmaskingScheduler = None, score_type: str = "confidence", early_stopping: bool = False, random_selection: bool = False):
        super().__init__(unmasking_scheduler, score_type=score_type, early_stopping=early_stopping, random_selection=random_selection)
        self.guidance_scale = guidance_scale  # CFG weight w
        self.remask_ratio = remask_ratio      # Re-mask ratio for uncond in each step ρ

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

        for i_step in range(timesteps):
            masked_index = (x[:, prompt_len:] == mask_token_id)
            if masked_index.sum() == 0:
                break
            
            ##############
            # Compute Conditional logits
            ##############
            cond_logits = model(x).logits
            NFE += 1
            
            ##############
            # Compute Unconditional logits
            ##############
            # token-level confidence for non-MASK tokens
            non_mask_index = torch.cat((torch.ones_like(x[:, :prompt_len], dtype=torch.bool), ~masked_index), dim=1)
            probs = torch.softmax(cond_logits, dim=-1)
            confidence = torch.max(probs, dim=-1).values
            
            # construct dynamic unconditional input
            x_uncond = x.clone()
            for b in range(x.shape[0]):
                candidate_idx = torch.nonzero(non_mask_index[b], as_tuple=False).squeeze(1)
                candidate_idx = candidate_idx[candidate_idx >= prompt_len]
                num_remask = min(max(1, int(len(candidate_idx) * self.remask_ratio)), len(candidate_idx))
                if num_remask > 0:
                    scores = confidence[b, candidate_idx]
                    low_idx = torch.topk(-scores, num_remask).indices
                    re_mask_pos = candidate_idx[low_idx]
                    x_uncond[b, re_mask_pos] = tokenizer.mask_token_id
            # unconditional logits
            uncond_logits = model(x_uncond).logits
            NFE += 1

            guided_logits = uncond_logits + (self.guidance_scale + 1.0) * (cond_logits - uncond_logits)

            # sample from guided logits at mask positions
            gen_logits = guided_logits[:, prompt_len:]
            gen_confidence, gen_x = self.sample_tokens(gen_logits, generation_config.temperature, generation_config.top_p, generation_config.top_k)

            selected_index = self.unmasking_scheduler.get_transfer_indices(gen_confidence, masked_index, i_step)
            if selected_index.shape[0] > 0:
                x[selected_index[:, 0], prompt_len + selected_index[:, 1]] = gen_x[selected_index[:, 0], selected_index[:, 1]]

            if self.early_stopping:
                x = self.propagate_eot_token(x, tokenizer.endoftext_token_id, prompt_length=prompt_len)

            history.append(x.clone())

        return OpenDLMOutput(sequences=x, history=history, NFE=NFE)