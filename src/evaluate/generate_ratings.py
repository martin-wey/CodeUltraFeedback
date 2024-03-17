import json
import os
import re

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from transformers import set_seed

from src.templates import (
    single_grading_system_prompt,
    single_grading_templates,
)
from src.utils import (
    H4ArgumentParser, ModelArguments, DataArguments, GenerationConfig,
    setup_logger, load_dataset, load_model_and_tokenizer
)

judges = ("gpt-3.5-turbo", "gpt-4-turbo", "claude-3-haiku-20240307")


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, GenerationConfig))
    model_args, data_args, generation_args = parser.parse()
    set_seed(42)

    logger = setup_logger()

    logger.info(f"Loading dataset: {data_args.dataset_name_or_path}")
    dataset = load_dataset(data_args.dataset_name_or_path, data_args.dataset_split)

    model_references = [column.split("_")[0] for column in dataset.column_names
                        if column not in ["instruction", "preference"]]
    if model_args.model_reference not in model_references:
        raise ValueError(f"{model_args.model_reference} not found in dataset fields.")

    logger.info("Loading responses...")
    model_test_path = os.path.join(data_args.model_responses_dir, f'{model_args.model_test}_responses.jsonl')
    assert os.path.exists(model_test_path), f"Cannot load model responses: {model_test_path} does not exist."

    with open(model_test_path, 'r') as f:
        responses = [json.loads(l) for l in f]

    logger.info(f"Loading judge model: {model_args.model_judge}")
    if model_args.model_judge not in judges:
        raise ValueError(f"{model_args.model_judge} not in judges: {judges}")

    generator, _, _ = load_model_and_tokenizer(model_args, generation_args)

    output_dir = os.path.join(data_args.model_responses_dir, 'single', model_args.model_reference)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{model_args.model_test}_judge_{model_args.model_judge}.jsonl')

    with open(output_file, "w") as fout:
        progress_bar_prefix = "Single-answer grading"
        with (Progress(
                TextColumn(
                    f"{progress_bar_prefix} - {model_args.model_judge} •" + "[progress.percentage]{task.percentage:>3.0f}%"
                ),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
        ) as p):
            for i, sample in p.track(enumerate(dataset), total=len(dataset)):
                # single-answer grading
                #
                # The LLM judge responds with a rating on a 1-10 scale and a rationale for the rating.
                #   The assessed LLM responses can potentially trigger `openai.BadRequestError`. It typically
                #   happens when the response contains hallucination in the form of repeated tokens/words.

                prompt = single_grading_templates[sample['preference']].format(
                    instruction=sample['instruction'],
                    reference=sample[f"{model_args.model_judge}_response"],
                    response=responses[i]
                )

                response = generator(
                    system_prompt=single_grading_system_prompt,
                    user_prompt=prompt
                )
                match = re.search(r'\bRating:\s*(\d+)\b', response)
                if match:
                    rating = int(match.group(1))
                else:
                    logger.info(response)
                    rating = "N/A"

                logger.info(f'Rating: {rating}')
                json.dump({'judgement': response, 'rating': rating}, fout)
                fout.write('\n')


if __name__ == "__main__":
    main()
