""" Example handler file. """

import runpod
import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer, TextGenerationPipeline

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.
device = "cuda:0"

model = AutoGPTQForCausalLM.from_quantized("/src/Mistral-7B-v0.1-qagen-v0.6-4bit")
tokenizer = AutoTokenizer.from_pretrained("/src/Mistral-7B-v0.1-qagen-v0.6-4bit", use_fast=True)


pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)


def generate(kwargs):
    return (pipeline(kwargs)[0]["generated_text"])



def handler(job):
    """ Handler function that will be used to process jobs. """
    params = job['input']


    output = generate(params)

    return output


runpod.serverless.start({"handler": handler})
