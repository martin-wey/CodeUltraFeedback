import json
import os
import random

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from transformers import set_seed

from src.templates import model_templates, principles
from src.utils import (
    H4ArgumentParser,
    ModelArguments, DataArguments, GenerationConfig,
    setup_logger, load_dataset, load_model_and_tokenizer,
    APICaller
)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, GenerationConfig))
    model_args, data_args, generation_args = parser.parse()
    set_seed(42)

    logger = setup_logger()

    logger.info(f"Loading dataset - {data_args.dataset_name_or_path}")
    dataset = load_dataset(data_args.dataset_name_or_path, data_args.dataset_split)
    logger.info(dataset)

    logger.info(f"Filtering dataset for model - {model_args.model_name}.")
    dataset = dataset.filter(lambda ex: model_args.model_name in ex['models'])

    if "gpt-" not in model_args.model_name:
        dataset = dataset.map(lambda ex: {
            'prompt': model_templates[model_args.model_name].template.format(
                principle=random.choice(principles[ex['preference']]),
                instruction=ex['instruction'])
        })

    logger.info(f"Loading model - {model_args.model_name}")
    generator, _, _ = load_model_and_tokenizer(model_args, generation_args)

    if not os.path.exists(data_args.output_dir):
        os.makedirs(data_args.output_dir, exist_ok=True)

    if not isinstance(generator, APICaller):
        generator_kwargs = {
            "max_new_tokens": generation_args.max_new_tokens,
            "do_sample": False,
            "return_full_text": False
        }
        if not generation_args.greedy:
            generator_kwargs.update({
                "do_sample": True,
                "temperature": generation_args.temperature,
                "top_p": generation_args.top_p
            })

    with open(os.path.join(data_args.output_dir, f'{model_args.model_name}.jsonl'), "w") as f:
        with Progress(
                TextColumn(
                    f"Generate responses - {model_args.model_name} •" + "[progress.percentage]{task.percentage:>3.0f}%"
                ),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
        ) as p:
            for sample in p.track(dataset, total=len(dataset)):
                if isinstance(generator, APICaller):
                    response = generator(
                        system_prompt=random.choice(principles[sample['preference']]),
                        user_prompt=sample['instruction'])
                else:
                    response = generator(sample['prompt'], **generator_kwargs)
                    response = response[0]['generated_text']

                json.dump({
                    'instruction': sample['instruction'],
                    'response': response
                }, f)
                f.write("\n")


if __name__ == "__main__":
    main()
