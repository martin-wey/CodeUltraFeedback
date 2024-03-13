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


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, GenerationConfig))
    model_args, data_args, generation_args = parser.parse()
    set_seed(42)

    logger = setup_logger()

    logger.info(f"Loading dataset: {data_args.dataset_name_or_path}")
    dataset = load_dataset(data_args.dataset_name_or_path, data_args.dataset_split)

    if data_args.references_path:
        logger.info(f"Loading references: {data_args.references_path}")
        assert os.path.exists(data_args.references_path), \
            f"Cannot load references file: {data_args.references_path} does not exist."
        with open(data_args.references_path, 'r') as f:
            references = [json.loads(l) for l in f]
        references_instructions = [r['instruction'] for r in references]

        # ensure the references are correctly mapped with the instructions
        def get_response(example):
            index = references_instructions.index(example['instruction'])
            return {'reference': references[index]['response']}

        dataset = dataset.add_column('reference', [None] * len(dataset))
        dataset = dataset.map(get_response)

    logger.info("Loading responses...")
    model_a_path = os.path.join(data_args.model_responses_dir, f'{model_args.model_a}_responses.jsonl')
    assert os.path.exists(model_a_path), f"Cannot load model responses: {model_a_path} does not exist."

    responses_data = []
    with open(model_a_path, 'r') as f:
        for line in f:
            line_data = json.loads(line)
            responses_data.append(line_data['response'])

    logger.info(f"Loading model: {model_args.model_name}")
    generator, _, _ = load_model_and_tokenizer(model_args, generation_args)

    output_dir = os.path.join(data_args.model_responses_dir, 'single')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{model_args.model_a}_judge_{model_args.model_judge}.jsonl')

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

                responses = responses_data[0]
                prompt = single_grading_templates[sample['preference']].format(
                    instruction=sample['instruction'],
                    reference=sample['reference'],
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
