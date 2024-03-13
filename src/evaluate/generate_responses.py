import json
import os

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from transformers import set_seed

from src.templates import principles
from src.utils import (
    H4ArgumentParser, ModelArguments, DataArguments, GenerationConfig,
    setup_logger, load_dataset, load_model_and_tokenizer, APICaller
)

if __name__ == '__main__':
    parser = H4ArgumentParser((ModelArguments, DataArguments, GenerationConfig))
    model_args, data_args, generation_args = parser.parse()
    set_seed(42)

    logger = setup_logger()

    logger.info(f"Loading dataset: {data_args.dataset_name_or_path}")
    dataset = load_dataset(data_args.dataset_name_or_path, data_args.dataset_split)
    column_names = list(dataset.features)

    logger.info(f"Loading model: {model_args.model_name}")
    generator, model, tokenizer = load_model_and_tokenizer(model_args, generation_args)
    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template

    if not isinstance(generator, APICaller):
        def apply_chat_template(example):
            prompt = [
                {'content': "", 'role': 'system'},
                {'content': f"{principles[example['preference']][0]}\n\n{example['instruction']}", 'role': 'user'}
            ]
            example['prompt'] = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            return example

        dataset = dataset.map(apply_chat_template)

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

    base_dir = os.path.join(data_args.output_dir, 'greedy' if generation_args.greedy else 'sampling')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    with open(os.path.join(base_dir, f'{model_args.model_name}_responses.jsonl'), "w") as f:
        with Progress(
            TextColumn(
                f"Generate responses  •" + "[progress.percentage]{task.percentage:>3.0f}%"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        ) as p:
            for sample in p.track(dataset, total=len(dataset)):
                if isinstance(generator, APICaller):
                    response = generator(
                        system_prompt="",
                        user_prompt=f"{principles[sample['preference']][0]}\n\n{sample['instruction']}")
                    response = response.choices[0].message.content
                    logger.info(f'Prompt:\n\n{sample["instruction"]}\n\nResponse:\n\n{response}')
                else:
                    response = generator(sample['prompt'], **generator_kwargs)
                    response = response[0]['generated_text']
                    logger.info(f'Prompt:\n\n{sample["prompt"]}\n\nResponse:\n\n{response}')
                json.dump({
                    'instruction': sample['instruction'],
                    'response': response
                }, f)
                f.write("\n")
