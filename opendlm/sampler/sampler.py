# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


from opendlm.model import LMGenerationConfig, OpenDLMOutput
from .scheduler import UnmaskingScheduler


    
class Sampler:
    """
    Base class for all samplers.
    
    Args:
        unmasking_scheduler: The unmasking scheduler.
        score_type: The score type for selecting tokens.
        early_stopping: If set to True, all subsequent positions after the end-of-text token will be filled with the end-of-text token. 
        random_selection: If set to True, the tokens will be selected randomly rather than using the score.
        
    Reference:
        - https://huggingface.co/Dream-org/Dream-v0-Instruct-7B
        - Nie, Shen, et al. "Large language diffusion models." arXiv preprint arXiv:2502.09992 (2025).
    """
    def __init__(self, unmasking_scheduler: UnmaskingScheduler = UnmaskingScheduler, score_type: str = "confidence", early_stopping: bool = False, random_selection: bool = False):
        
        if score_type not in ["confidence", "margin", "entropy"]:
            raise ValueError(f"Invalid score type: {score_type}")
        
        self.unmasking_scheduler = unmasking_scheduler
        self.score_type = score_type
        self.random_selection = random_selection
        self.early_stopping = early_stopping
        

    def generate(self, input_ids, attention_mask, model: AutoModel, tokenizer: AutoTokenizer, generation_config: LMGenerationConfig):
        NFE = 0
        history = []
        gen_length = generation_config.max_new_tokens
        timesteps = generation_config.timesteps
        mask_token_id = tokenizer.mask_token_id
        endoftext_token_id = tokenizer.endoftext_token_id
        
        # if attention_mask is not None and torch.any(attention_mask == 0.0):
        #     # we do not mask the [MASK] tokens so value = 1.0
        #     attention_mask = F.pad(attention_mask, (0, gen_length), value=1.0)
        #     tok_idx = attention_mask.long().cumsum(-1) - 1
        #     tok_idx.masked_fill_(attention_mask == 0, 1)
        #     # attention_mask is of shape [B, N]
        #     # broadcast to [B, 1, N, N]
        #     ori_attention_mask = torch.logical_and(
        #         attention_mask.unsqueeze(1).unsqueeze(-2),
        #         attention_mask.unsqueeze(1).unsqueeze(-1),
        #     )
        # else:
        #     tok_idx = torch.arange(input_ids.shape[1]+gen_length, device=input_ids.device
        #     ).unsqueeze(0).repeat(input_ids.shape[0], 1)
        #     attention_mask = "full"
        #     ori_attention_mask = "full"
        
        # Shape: [B, L]
        x = torch.full((input_ids.shape[0], input_ids.shape[1] + gen_length), mask_token_id, dtype=torch.long).to(input_ids.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        
        for i_step in range(timesteps):
            # Check the mask index of the generation tokens
            masked_index = (x[:, input_ids.shape[1]:] == mask_token_id)

            if masked_index.sum() == 0:
                break
            # the logits of next time step
            logits = model(x).logits
            NFE += 1
            gen_logits = logits[:, input_ids.shape[1]:]
            
            # sample the tokens
            gen_confidence, gen_x = self.sample_tokens(gen_logits, generation_config.temperature, generation_config.top_p, generation_config.top_k)
            
            selected_index = self.unmasking_scheduler.get_transfer_indices(gen_confidence, masked_index, i_step)
            if selected_index.shape[0] > 0:
                x[selected_index[:, 0], input_ids.shape[1] + selected_index[:, 1]] = gen_x[selected_index[:, 0], selected_index[:, 1]]
            

            if self.early_stopping:
                x = self.propagate_eot_token(x, endoftext_token_id, prompt_length=input_ids.shape[1])
                            
            history.append(x.clone())

        return OpenDLMOutput(sequences=x, history=history, NFE=NFE)
    
    def propagate_eot_token(self, x, endoftext_id, prompt_length):
        for sample_idx in range(x.shape[0]):
            if endoftext_id in x[sample_idx, prompt_length:]: # endoftext token
                pad_positions = (x[sample_idx, prompt_length:] == endoftext_id).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    first_pad_pos = pad_positions[0]
                    absolute_pad_pos = prompt_length + first_pad_pos
                    x[sample_idx, absolute_pad_pos:] = endoftext_id
        return x
    
    def sample_tokens(self, logits, temperature=0.0, top_p=None, top_k=None):
        
        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1:
            logits = self.top_p_logits(logits, top_p)
        if top_k is not None and top_k > 0:
            logits = self.top_k_logits(logits, top_k)
        probs = torch.softmax(logits, dim=-1)

        if temperature > 0:
            try:
                x0 = dists.Categorical(probs=probs).sample()
                confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            except:
                confidence, x0 = probs.max(dim=-1)
        else:
            confidence, x0 = probs.max(dim=-1)
            
        if self.random_selection:
            confidence = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            if self.score_type == "margin":
                sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
                # Extract top1 and top2 probabilities
                top1_probs = sorted_probs[:, 0] 
                top2_probs = sorted_probs[:, 1] 
                # Calculate confidence as top1 - top2
                confidence = top1_probs - top2_probs 
            
            if self.score_type == "entropy":
                epsilon = 1e-10
                log_probs = torch.log(probs + epsilon)
                confidence = torch.sum(probs * log_probs, dim=-1)
                
        return confidence, x0
    
    
    def top_p_logits(self, logits, top_p=None):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
        mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        return logits

    def top_k_logits(self, logits, top_k=None):
        top_k = min(top_k, logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
        return logits
    
    def transfer_tokens(self, x, mask_index, mask_x, mask_transfer_index):
        masked_pos = torch.nonzero(mask_index, as_tuple=False)  # shape: [num_masked, 2]
        selected_pos = masked_pos[mask_transfer_index]  # shape: [num_selected, 2]
        x[selected_pos[:, 0], selected_pos[:, 1]] = mask_x[mask_transfer_index]
        return x
    


