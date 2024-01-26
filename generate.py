import argparse

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Tokenizer,
    model_init,
)
from datasets import load_from_disk
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

from prompt_templates import templates, system_message

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_dir", type=str, help="Path to model directory")
    parser.add_argument("-d", "--dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument("-mt", "--model_type", type=str)
    parser.add_argument("-b", "--batch_size", default=5, type=int)
    parser.add_argument("-s", "--seed", default=42, type=int)
    # model_init.add_args(parser)
    args = parser.parse_args()

    print(" -- Loading dataset: " + args.dataset_dir)
    dataset = load_from_disk(args.dataset_dir)
    instructions = dataset["instruction"]
    s_instructions = sorted(instructions, key=len)

    print(" -- Batching dataset...")
    prompts = [templates['codellama_instruct'].format(system_message=system_message,
                                                      instruction=i) for i in s_instructions]
    batches = [prompts[i:i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]

    config = ExLlamaV2Config()
    config.model_dir = args.model_dir
    config.prepare()

    config.max_batch_size = args.batch_size

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
    for b, batch in enumerate(batches):
        print(f"Batch {b + 1} of {len(batches)}...")

        outputs = generator.generate_simple(batch, settings, max_new_tokens, seed=1234)

        trimmed_outputs = [o[len(p):] for p, o in zip(batch, outputs)]
        collected_outputs += trimmed_outputs

    for q, a in zip(s_instructions, collected_outputs):
        print("---------------------------------------")
        print("Q: " + q)
        print("A: " + a)
