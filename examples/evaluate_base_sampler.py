import torch

from opendlm.model import OpenDLM, LMGenerationConfig
from opendlm.sampler import Sampler, AdaptiveCFGSampler, NullCFGSampler
from opendlm.sampler.scheduler import LinearUnmaskingScheduler, DetailedUnmaskingScheduler

model_name = "Dream-org/Dream-v0-Instruct-7B"
open_dlm = OpenDLM(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
open_dlm.model = open_dlm.model.to("cuda").eval()

unmasking_scheduler = LinearUnmaskingScheduler(gen_length=256, timesteps=256)
# unmasking_scheduler = DetailedUnmaskingScheduler(gen_length=256, alpha=2.0)


# Define the generation config, including the max new tokens, timesteps, temperature, top_p, top_k
llm_generation_config = LMGenerationConfig(max_new_tokens=256, timesteps=256, temperature=0.2, top_p=0.95, top_k=20)
# Choose the sampler
sampler = Sampler(unmasking_scheduler=unmasking_scheduler, score_type="confidence", propagate_eot=False, random_selection=False)

# The null classifier-free guidance sampler
# sampler = NullCFGSampler(unmasking_scheduler=unmasking_scheduler, score_type="confidence", propagate_eot=False, random_selection=False)

# The adaptive classifier-free guidance sampler
# sampler = AdaptiveCFGSampler(unmasking_scheduler=unmasking_scheduler, score_type="confidence", propagate_eot=False, random_selection=False)



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








