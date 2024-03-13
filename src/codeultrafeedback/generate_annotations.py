import json
import os
import random
import re

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
from transformers import set_seed

from src.templates import judge_system_prompt, judge_templates
from src.utils import (
    H4ArgumentParser, ModelArguments, DataArguments, GenerationConfig,
    setup_logger, load_dataset, load_model_and_tokenizer
)


# Adapted from https://github.com/OpenBMB/UltraFeedback/blob/main/src/data_annotation/annotate_preference.py#L18
def process(responses):
    annotations = {'ratings': [], 'rationales': []}
    responses = responses.split("\n\n")
    if len(responses) != 4:
        annotations['rationales'] = responses
        return annotations
    try:
        pattern = r'Rating: (.+?)\nRationale: (.+)'
        for response in responses:
            matches = re.search(pattern, response, re.DOTALL)
            try:
                annotations['ratings'].append(
                    re.findall(r'\b\d+\b', matches.group(1))[0] if matches.group(1) != 'N/A' else 'N/A')
            except Exception:
                raise Exception
            annotations['rationales'].append(matches.group(2))
    except ValueError or AttributeError as e:
        annotations['rationales'] = responses
        return annotations
    except IndexError as e:
        print(responses)
        raise IndexError
    return annotations


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, GenerationConfig))
    model_args, data_args, generation_args = parser.parse()
    set_seed(42)

    logger = setup_logger()

    logger.info(f"Loading dataset - {data_args.dataset_name_or_path}")
    dataset = load_dataset(data_args.dataset_name_or_path, data_args.dataset_split)
    logger.info(dataset)

    logger.info(f"Loading model - {model_args.model_name}")
    generator, _, _ = load_model_and_tokenizer(model_args, generation_args)

    if not os.path.exists(data_args.output_dir):
        os.makedirs(data_args.output_dir, exist_ok=True)

    with open(os.path.join(data_args.output_dir, f'annotations_{data_args.model_name}.jsonl'), "w") as f:
        with Progress(
                TextColumn(
                    f"Generate annotations - {model_args.model_name} •" + "[progress.percentage]{task.percentage:>3.0f}%"
                ),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
        ) as p:
            for sample in p.track(dataset, total=len(dataset)):
                format_input = {'instruction': sample['instruction']}
                order = list(range(4))
                random.shuffle(order)
                format_input.update({
                    f'response_{i + 1}': sample['responses'][o]['response'] for i, o in enumerate(order)
                })

                response = generator(
                    system_prompt=judge_system_prompt,
                    user_prompt=judge_templates[sample['preference']].format(**format_input))
                if response is not None:
                    annotations = process(response)
                    annotations['order'] = order
                else:
                    annotations = {'ratings': [], 'rationales': [], 'order': order}
                json.dump(annotations, f)
                f.write('\n')


if __name__ == "__main__":
    main()
