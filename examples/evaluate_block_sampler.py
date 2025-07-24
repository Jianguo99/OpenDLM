import torch

from opendlm.model import OpenDLM, LMGenerationConfig
from opendlm.sampler import BlockSampler
from opendlm.sampler.scheduler import FlexibleUnmaskingScheduler, DetailedUnmaskingScheduler

model_name = "GSAI-ML/LLaDA-8B-Instruct"
open_dlm = OpenDLM(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
open_dlm.model = open_dlm.model.to("cuda").eval()
# Define the generation config, including the max new tokens, timesteps, temperature, top_p, top_k
llm_generation_config = LMGenerationConfig(max_new_tokens=256, timesteps=256, temperature=0.0, top_p=1.0, top_k=0)
# Choose the sampler
base_scheduler = FlexibleUnmaskingScheduler(gen_length=256, timesteps=256, schedule_type="linear")
# base_scheduler = DetailedUnmaskingScheduler(gen_length=32, alpha=2.0)
sampler = BlockSampler(unmasking_scheduler=base_scheduler, score_type="confidence", propagate_eot=False, random_selection=False, block_length=32)

# Define the system prompt and messages
system_prompt = "Please reason step by step, and put your final answer within \\boxed{{}}."
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"}
]
inputs = open_dlm.tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True, padding=True
)
# Move the inputs to the GPU
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")
output = open_dlm.generate(sampler=sampler, input_ids=input_ids, attention_mask=attention_mask, return_dict=True, generation_config=llm_generation_config)
print("Response:")
# Get endoftext token
endoftext_token = open_dlm.tokenizer.decode(open_dlm.endoftext_id)
responses = [open_dlm.tokenizer.decode(sequence[len(input_ids[0]):]).split(endoftext_token)[0] for sequence in output.sequences]
print(responses)
print("NFE:")
print(output.NFE)








