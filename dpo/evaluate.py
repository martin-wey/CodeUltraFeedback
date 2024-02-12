import argparse
import logging
import random

import torch
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from transformers import set_seed, pipeline, AutoTokenizer

from principles import principles

logger = logging.getLogger("rich")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, help="Path to model directory")
    parser.add_argument("-d", "--dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to output model responses")
    parser.add_argument("-s", "--seed", default=42, type=int)
    args = parser.parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )

    print(" -- Loading dataset: " + args.dataset_dir)
    dataset = load_from_disk(args.dataset_dir)["test"]

    print(" -- Loading model: " + args.model_dir)
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        use_cache=True,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    column_names = list(dataset.features)

    def apply_chat_template(example):
        prompt_msg = [
            {'content': random.choice(principles[example['preference']]), 'role': 'system'},
            {'content': example['instruction'], 'role': 'user'},
        ]
        example['prompt'] = tokenizer.apply_chat_template(prompt_msg, tokenize=False)

        return example

    dataset = dataset.map(
        apply_chat_template,
        remove_columns=column_names,
        desc="Applying chat template"
    )

    with (Progress(
            TextColumn(
                f"Generate responses •" + "[progress.percentage]{task.percentage:>3.0f}%"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
    ) as p):
        for sample in p.track(dataset.select(range(3)), total=len(dataset)):
            response = generator(sample['prompt'],
                                 max_new_tokens=1024,
                                 do_sample=True,
                                 temperature=0.8,
                                 top_k=50,
                                 top_p=0.8,
                                 return_full_text=False)
            response = response[0]['generated_text']
            logger.info(f"Prompt sample:\n\n{sample['prompt']}\n\nResponse:\n\n{response}")