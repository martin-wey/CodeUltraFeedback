import argparse
import json
import os
import random

from auto_gptq import AutoGPTQForCausalLM
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

from prompt_templates import system_messages, principles, templates, system_mappings
from utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, help="Path to model directory")
    parser.add_argument("-d", "--dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to output model responses")
    parser.add_argument("-mn", "--model_name", type=str)
    parser.add_argument("-s", "--seed", default=42, type=int)
    args = parser.parse_args()
    set_seed(args.seed)

    print(" -- Loading model: " + args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(args.model_dir, device_map="auto")

    print(" -- Loading dataset: " + args.dataset_dir)
    dataset = load_from_disk(args.dataset_dir)
    dataset = dataset.filter(lambda ex: args.model_name in ex['models'])
    print(f" -- Number of samples for {args.model_name}: {len(dataset)}")
    dataset = dataset.map(lambda ex: {
        'prompt': templates[args.model_name].format(system_message=system_messages[system_mappings[args.model_name]],
                                                    principle=random.choice(principles[ex['preference']]),
                                                    instruction=ex['instruction'])
    })

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    with open(os.path.join(args.output_dir, f"{args.model_name}.jsonl"), "w") as f:
        for b, sample in tqdm(enumerate(dataset), total=len(dataset)):
            output = pipe(sample['prompt'], max_new_tokens=1024, do_sample=True, temperature=0.8, top_k=50, top_p=0.95)
            trimmed_output = output[0]['generated_text'][len(sample['prompt']):]

            json.dump({
                'instruction': sample['instruction'],
                'response': trimmed_output
            }, f)
            f.write("\n")
