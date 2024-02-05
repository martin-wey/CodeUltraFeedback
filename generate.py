import argparse
import json
import os
import random
import time

import torch
from auto_gptq import AutoGPTQForCausalLM
from datasets import load_from_disk
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, set_seed

from templates import principles, templates


# set OPENAI_API_KEY env. variable with OpenAI key

class APICaller:
    def __init__(self, model, client):
        self.model = model
        self.client = client

    def __call__(self, system_prompt, user_prompt):
        content = None
        for _ in range(20):
            try:
                response = self.client.chat.completions.create(**{
                    'model': self.model,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'temperature': 0.8,
                    'max_tokens': 1024,
                    'top_p': 0.8,
                })
                content = response.choices[0].message.content
            except Exception as e:
                print(e)
                time.sleep(1)
            else:
                break
        return content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, help="Path to model directory")
    parser.add_argument("-d", "--dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to output model responses")
    parser.add_argument("-st", "--start", default=0, type=int)
    parser.add_argument("-mn", "--model_name", type=str)
    parser.add_argument("-s", "--seed", default=42, type=int)
    args = parser.parse_args()
    set_seed(args.seed)

    print(" -- Loading dataset: " + args.dataset_dir)
    dataset = load_from_disk(args.dataset_dir)
    dataset = dataset.filter(lambda ex: args.model_name in ex['models'])
    print(f" -- Number of samples for {args.model_name}: {len(dataset)}")
    if args.model_name not in ['gpt-3.5-turbo', 'gpt-4']:
        dataset = dataset.map(lambda ex: {
            'prompt': templates[args.model_name].template.format(
                principle=random.choice(principles[ex['preference']]),
                instruction=ex['instruction'])
        })

    if args.start > 0:
        dataset = dataset.select(range(args.start, len(dataset)))
        print(dataset)

    if args.model_name in ['gpt-3.5-turbo', 'gpt-4']:
        if args.model_name == 'gpt-4':
            model = 'gpt-4-turbo-preview'
        else:
            model = 'gpt-3.5-turbo-0125'
        client = OpenAI()
        generator = APICaller(model, client)
    else:
        print(" -- Loading model: " + args.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
        model = AutoGPTQForCausalLM.from_quantized(args.model_dir, device_map='auto', torch_dtype=torch.float16)
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    if args.start > 0:
        file_name = f'{args.model_name}_{args.start}.jsonl'
    else:
        file_name = f'{args.model_name}.jsonl'

    with open(os.path.join(args.output_dir, file_name), "w") as f:
        for b, sample in tqdm(enumerate(dataset), total=len(dataset)):
            if args.model_name in ['gpt-3.5-turbo', 'gpt-4']:
                response = generator(system_prompt=random.choice(principles[sample['preference']]),
                                     user_prompt=sample['instruction'])
            else:
                response = generator(sample['prompt'],
                                     max_new_tokens=1024,
                                     do_sample=True,
                                     temperature=0.8,
                                     top_k=50,
                                     top_p=0.8)
                response = response[0]['generated_text'][len(sample['prompt']):]

            json.dump({
                'instruction': sample['instruction'],
                'response': response
            }, f)
            f.write("\n")
