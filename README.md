**OpenDLM** is an open-source library focused on sampling algorithms for Diffusion Language Models (DLMs). It offers a modular and extensible framework for implementing, evaluating, and benchmarking a wide range of sampling strategies for multi-step generation.
Currently under active development, this project is maintained by [An Bo's research group](https://personal.ntu.edu.sg/boan/) at Nanyang Technological University (NTU). We warmly welcome feedback, issues, contributions, and collaborative opportunities from the community!

## 🚀 Overview of OpenDLM

OpenDLM currently supports a growing set of Diffusion Language Models (DLMs) and advanced sampling strategies:

| ✅ **Feature**               | **Details**                                                                                                                                                                                                                                               |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Supported DLMs**          | - [LLaDA: Large Language Diffusion Models](https://github.com/ML-GSAI/LLaDA) <br> - [Dream 7B](https://github.com/DreamLM/Dream)                                                                                                                            |
| **Sampling Strategies**     | - **Base Sampler** <br> - **Block Sampler** <br> - **Null Classifier-guided sampling** <br> - **Adaptive Classifier-guided sampling** <br> |
| **Unmasking Schedulers**     | - **Linear Scheduler** <br> - **Detailed Unmasking Schedulers** |

---

## 🧩 TODO List  
We welcome your contributions! Items marked **High Priority**</span> are especially encouraged.

| 🔧 **Feature**              | 🛠️ **Planned Work**                                                                                                   |
|----------------------------|------------------------------------------------------------------------------------------------------------------------|
| **More DLMs**              | - [d1](https://github.com/dllm-reasoning/d1)        |
| **Sampling Strategies**    | - (New strategy contributions are welcome!)         |
| **RL Training**            | - Implement GRPO-based reinforcement learning       |





## Installation

OpenDLM is developed with Python 3.10 and PyTorch 2.4.0+cu121. To install OpenDLM, simply run

```
pip install opendlm
```

## Examples

Here, we provide a simple example for OpenDLM.

```python
import torch

from opendlm.model import OpenDLM, LMGenerationConfig
from opendlm.sampler import Sampler, AdaptiveCFGSampler, NullCFGSampler
from opendlm.sampler.scheduler import LinearUnmaskingScheduler, DetailedUnmaskingScheduler

model_name = "Dream-org/Dream-v0-Instruct-7B"
open_dlm = OpenDLM(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
open_dlm.model = open_dlm.model.to("cuda").eval()

unmasking_scheduler = LinearUnmaskingScheduler(gen_length=256, timesteps=256)
# Define the generation config, including the max new tokens, timesteps, temperature, top_p, top_k
llm_generation_config = LMGenerationConfig(max_new_tokens=256, timesteps=256, temperature=0.2, top_p=0.95, top_k=20)
# Choose the sampler
sampler = Sampler(unmasking_scheduler=unmasking_scheduler, score_type="confidence", propagate_eot=False, random_selection=False)

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
```

You may find more tutorials in [`examples`](https://github.com/Jianguo99/OpenDLM/examples) folder.

## License

This project is licensed under the MIT. The terms and conditions can be found in the LICENSE file.

## Contributors

* [Jianguo Huang](https://jianguo99.github.io/)

[contributors-shield]: https://img.shields.io/github/contributors/Jianguo99/OpenDLM.svg?style=for-the-badge

[contributors-url]: https://github.com/Jianguo99/OpenDLM/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/Jianguo99/OpenDLM.svg?style=for-the-badge

[forks-url]: https://github.com/Jianguo99/OpenDLM/network/members

[stars-shield]: https://img.shields.io/github/stars/Jianguo99/OpenDLM.svg?style=for-the-badge

[stars-url]: https://github.com/Jianguo99/OpenDLM/stargazers

[issues-shield]: https://img.shields.io/github/issues/Jianguo99/OpenDLM.svg?style=for-the-badge

[issues-url]: https://github.com/Jianguo99/OpenDLM/issues

[license-shield]: https://img.shields.io/github/license/Jianguo99/OpenDLM.svg?style=for-the-badge

[license-url]: https://github.com/Jianguo99/OpenDLM/blob/main/LICENSE.txt

[tag-shield]: https://img.shields.io/github/v/tag/Jianguo99/OpenDLM?style=for-the-badge&label=version

[tag-url]: https://github.com/Jianguo99/OpenDLM/tags

