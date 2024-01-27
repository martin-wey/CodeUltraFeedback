import argparse
import json
import os
import random

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
    parser.add_argument("-b", "--batch_size", default=4, type=int)
    parser.add_argument("-s", "--seed", default=42, type=int)
    model_init.add_args(parser)
    args = parser.parse_args()
    set_seed(args.seed)

    config = ExLlamaV2Config()
    config.model_dir = args.model_dir
    config.prepare()
    config.max_batch_size = args.batch_size

    print(" -- Loading dataset: " + args.dataset_dir)
    dataset = load_from_disk(args.dataset_dir)
    dataset = dataset.filter(lambda e: args.model_name in e['models'])
    print(f" -- Number of samples for {args.model_name}: {len(dataset)}")
    instructions = dataset["instruction"]
    preferences = dataset['preference']

    # sort the instructions by length for more efficient batched padding
    ip_pairs = list(zip(instructions, preferences))
    sorted_ip_pairs = sorted(ip_pairs, key=lambda p: len(p[0]))
    s_instructions, s_preferences = zip(*sorted_ip_pairs)

    print(" -- Batching dataset...")
    prompts = [templates[args.model_name].format(system_message=system_messages[system_mappings[args.model_name]],
                                                 principle=random.choice(principles[p]),
                                                 instruction=i) for i, p in sorted_ip_pairs]
    batches = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    model = ExLlamaV2(config)
    print(" -- Loading model: " + args.model_dir)
    cache = ExLlamaV2Cache(model, lazy=True, batch_size=args.batch_size)  # Cache needs to accommodate the batch size
    model.load_autosplit(cache)
    tokenizer = ExLlamaV2Tokenizer(config)
    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.85
    settings.top_k = 50
    settings.top_p = 0.8
    settings.token_repetition_penalty = 1.05
    max_new_tokens = 1024

    collected_outputs = []
    with open(os.path.join(args.output_dir, f"{args.model_name}.jsonl"), "w") as f:
        with tqdm(total=len(batches)) as pbar:
            for b, batch in enumerate(batches):
                pbar.set_description(f"Batch {b}/{len(batches)}")

                outputs = generator.generate_simple(batch, settings, max_new_tokens, seed=args.seed)

                trimmed_outputs = [o[len(p):] for p, o in zip(batch, outputs)]
                collected_outputs += trimmed_outputs

                batch_instructions = s_instructions[b * len(batches):(b+1) * len(batches)]

                for i, o in zip(batch_instructions, trimmed_outputs):
                    json.dump({"instruction": i, "response": o}, f)
                    f.write("\n")

                pbar.update(1)
