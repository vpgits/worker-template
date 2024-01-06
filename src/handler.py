""" Example handler file. """

import runpod
import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

model = AutoGPTQForCausalLM.from_quantized("/src/Mistral-7B-v0.1-qagen-v0.6-4bit")
tokenizer = AutoTokenizer.from_pretrained("/src/Mistral-7B-v0.1-qagen-v0.6-4bit", use_fast=True)



def generate(prompt, token_limit=512):
    model.eval()
    with torch.no_grad():
        model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
        return tokenizer.decode(model.generate(**model_input, max_new_tokens=token_limit)[0], skip_special_tokens=True)



def handler(job):
    """ Handler function that will be used to process jobs. """
    prompt = job['input']['text']
    token_limit = job['input']['token_limit']



    output = generate(prompt, token_limit)

    return output


runpod.serverless.start({"handler": handler})
