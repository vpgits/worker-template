""" Example handler file. """

import runpod
import torch
from auto_gptq import AutoGPTQForCausalLM, TextGenerationPipeline
from transformers import AutoTokenizer

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

model = AutoGPTQForCausalLM.from_quantized("src/Mistral-7B-v0.1-qagen-v0.6-4bit")
tokenizer = AutoTokenizer.from_pretrained("src/Mistral-7B-v0.1-qagen-v0.6-4bit", use_fast=True)



def generate(prompt):
    return (tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(model.device))[0]))



def handler(job):
    """ Handler function that will be used to process jobs. """
    prompt = job['input']

    output = generate(prompt)

    return output


runpod.serverless.start({"handler": handler})
