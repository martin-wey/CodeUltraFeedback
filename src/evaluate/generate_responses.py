import argparse
import json
import logging
import os

import torch
from auto_gptq import AutoGPTQForCausalLM
from datasets import load_from_disk
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from transformers import set_seed, pipeline, AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel

from src.templates import principles, model_templates

logger = logging.getLogger("rich")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--output_dir", type=str, default="./runs/eval_results/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )

    logging.info("Loading dataset: " + args.dataset_dir)
    dataset = load_from_disk(args.dataset_dir)["test"]
    column_names = list(dataset.features)

    logging.info("Loading model: " + args.model_name_or_path)
    if "-sft" in args.model_name or "-dpo" in args.model_name:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name_or_path,
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
        tokenizer.chat_template = (
            "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' "
            "+ message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ "
            "'<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == "
            "'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% "
            "endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% "
            "endif %}\n{% endfor %}")

        def apply_chat_template(example):
            # @todo: perhaps the principle should appear in the <|user|> part and not system
            # SFT -> in instruction
            # DPO -> in system prompt
            prompt_msg = [
                {'content': principles[example['preference']][0], 'role': 'system'},
                {'content': example['instruction'], 'role': 'user'},
            ]
            example['prompt'] = tokenizer.apply_chat_template(prompt_msg, tokenize=False)
            return example

        dataset = dataset.map(apply_chat_template)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            trust_remote_code=False,
            torch_dtype=torch.float16,
            revision=args.revision
        )
        dataset = dataset.map(lambda ex: {
            'prompt': model_templates[args.model_name].template.format(
                principle=principles[ex['preference']][0],
                instruction=ex['instruction'])
        })

    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    with open(os.path.join(args.output_dir, f'{args.model_name}_responses.jsonl'), "w") as f:
        with (Progress(
                TextColumn(f"Generate responses •" + "[progress.percentage]{task.percentage:>3.0f}%"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
        ) as p):
            for sample in p.track(dataset, total=len(dataset)):
                response = generator(
                    sample['prompt'],
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.8,
                    return_full_text=False
                )
                response = response[0]['generated_text']
                logger.info(f'Prompt:\n\n{sample["prompt"]}\n\nResponse:\n\n{response}')
                json.dump({
                    'instruction': sample['instruction'],
                    'response': response
                }, f)
                f.write("\n")
