# Copyright (c) 2025-present, AI-for-X, NTU.
# All rights reserved.
#
# Licensed under the license found in the LICENSE file in the root directory of this source tree.

import torch
from opendlm.model import LMGenerationConfig, OpenDLMOutput
from .block import BlockSampler
from .scheduler.eb import EntropyBoundedUnmaskingScheduler 

class EBSampler(BlockSampler):
    """
    EBSampler: Entropy-Bounded Unmasking (EB-Sampler)
    
    Reference:
        Ben-Hamu et al. "Accelerated Sampling from Masked Diffusion Models via Entropy Bounded Unmasking" (2025)
        https://arxiv.org/abs/2505.24857
    """

    def __init__(self, block_length: int = 32, gamma: float = 0.1, score_type="entropy", propagate_eot: bool = False):
        eb_scheduler = EntropyBoundedUnmaskingScheduler(gamma=gamma)
        super().__init__(block_length=block_length, unmasking_scheduler=eb_scheduler, 
                         score_type=score_type, propagate_eot=propagate_eot, random_selection=False)

    def generate(self, input_ids, attention_mask, model, tokenizer, generation_config: LMGenerationConfig):
        NFE = 0
        history = []
        gen_length = generation_config.max_new_tokens
        block_length = self.block_length
        timesteps = generation_config.timesteps
        mask_token_id = tokenizer.mask_token_id
        endoftext_token_id = tokenizer.endoftext_token_id

        # 检查分块一致性
        if gen_length % block_length != 0:
            raise ValueError(f"gen_length {gen_length} must be divisible by block_length {block_length}")
        if timesteps % (gen_length // block_length) != 0:
            raise ValueError(f"timesteps {timesteps} must be divisible by num_blocks {gen_length // block_length}")

        # 初始化 x
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
                if block_masked_index.sum() == 0:
                    break

                # 模型前向
                logits = model(x).logits
                NFE += 1
                gen_logits = logits[:, block_start:block_end]

                # 选取每个位置最可能token (argmax)
                gen_confidence, gen_entropy, gen_block_x = self.sample_tokens(gen_logits, generation_config.temperature, generation_config.top_p, generation_config.top_k)
               

                # 调用EB-Scheduler，选择当前step要unmask的token
                selected_index = self.unmasking_scheduler.get_transfer_indices(
                    gen_entropy, gen_confidence, block_masked_index, i_step
                )

                if selected_index.shape[0] > 0:
                    x[selected_index[:, 0], block_start + selected_index[:, 1]] = gen_block_x[selected_index[:, 0], selected_index[:, 1]]

                if self.propagate_eot:
                    x = self.propagate_eot_token(x, endoftext_token_id, prompt_length=input_ids.shape[1])

                history.append(x.clone())

        return OpenDLMOutput(sequences=x, history=history, NFE=NFE)


    def sample_tokens(self, logits, temperature=0.0, top_p=None, top_k=None):
        
        if temperature > 0:
            logits = logits / temperature
        if top_p is not None and top_p < 1:
            logits = self.top_p_logits(logits, top_p)
        if top_k is not None and top_k > 0:
            logits = self.top_k_logits(logits, top_k)
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate entropy
        entropy = -self.neg_entropy_score(probs)

        # Sample tokens
        if temperature > 0:
            try:
                x0 = torch.distributions.Categorical(probs=probs).sample()
                confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            except:
                confidence, x0 = probs.max(dim=-1)
        else:
            confidence, x0 = probs.max(dim=-1)
            
        if self.random_selection:
            confidence = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            if self.score_type == "margin":
                confidence = self.margin_score(probs)
            
            elif self.score_type == "entropy":
                confidence = self.neg_entropy_score(probs)
            elif self.score_type == "confidence":
                confidence = confidence
            else:
                raise ValueError(f"Invalid score_type: {self.score_type}")
                
        return confidence, entropy, x0