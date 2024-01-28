import argparse
import json
import os
import random

import torch
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    model_init,
)
from datasets import load_from_disk
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
from tqdm import tqdm

from prompt_templates import system_messages, principles, templates, system_mappings
from utils import set_seed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("-o", "--output_dir", type=str, help="Path to output model responses")
    parser.add_argument("-mn", "--model_name", type=str)
    parser.add_argument("-nwu", "--no_warmup", action="store_true", help="Skip warmup before testing model")
    parser.add_argument("-s", "--seed", default=42, type=int)
    model_init.add_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)

    model_init.check_args(args)
    model_init.print_options(args)
    model, tokenizer = model_init.init(args, allow_auto_split=True)
    cache = None

    if not model.loaded:
        print(" -- Loading model...")
        cache = ExLlamaV2Cache(model, lazy=True)
        model.load_autosplit(cache)

    print(" -- Loading dataset: " + args.dataset_dir)
    dataset = load_from_disk(args.dataset_dir)
    dataset = dataset.filter(lambda ex: args.model_name in ex['models'])
    print(f" -- Number of samples for {args.model_name}: {len(dataset)}")
    dataset = dataset.map(lambda ex: {
        'prompt': templates[args.model_name].format(system_message=system_messages[system_mappings[args.model_name]],
                                                    principle=random.choice(principles[ex['preference']]),
                                                    instruction=ex['instruction'])
    })

    with torch.inference_mode():
        if cache is None:
            cache = ExLlamaV2Cache(model)
        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        if not args.no_warmup:
            generator.warmup()

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.8
        settings.top_k = 50
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.05
        max_new_tokens = 1024

        with open(os.path.join(args.output_dir, f"{args.model_name}.jsonl"), "w") as f:
            for b, sample in tqdm(enumerate(dataset), total=len(dataset)):
                output = generator.generate_simple(sample['prompt'], settings, max_new_tokens, seed=args.seed)
                trimmed_output = output[len(sample['prompt']):]

                json.dump({
                    'instruction': sample['instruction'],
                    'response': trimmed_output
                }, f)
                f.write("\n")
