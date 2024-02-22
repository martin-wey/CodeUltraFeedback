import argparse
import json
import logging
import os
import random
import re

import openai
from datasets import load_from_disk, load_dataset
from openai import OpenAI
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from src.templates import (
    pairwise_grading_system_prompt,
    pairwise_grading_templates,
    single_grading_system_prompt,
    single_grading_templates
)

logger = logging.getLogger("rich")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to dataset directory.")
    parser.add_argument("--model_responses_dir", type=str, default="./runs/eval_results/")
    parser.add_argument("--model_a", type=str, required=True, help="Name of first model to judge.")
    parser.add_argument("--model_b", type=str, help="Name of second model to judge.")
    parser.add_argument("--model_judge", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--references_fp", type=str, required=True, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )

    logger.info("Loading test dataset: " + args.dataset_dir)
    dataset = load_from_disk(args.dataset_dir)["test"]

    logger.info("Loading references: " + args.references_fp)
    assert os.path.exists(args.references_fp), f"Cannot load references file: {args.references_fp} does not exist."
    with open(args.references_fp, 'r') as f:
        references = [json.loads(l) for l in f]
    references_instructions = [r['instruction'] for r in references]

    def get_response(example):
        index = references_instructions.index(example['instruction'])
        return {'reference': references[index]['response']}

    dataset = dataset.add_column('reference', [None] * len(dataset))
    dataset = dataset.map(get_response)

    logger.info("Loading responses...")
    model_a_path = os.path.join(args.model_responses_dir, f'{args.model_a}_responses.jsonl')
    assert os.path.exists(model_a_path), f"Cannot load model responses: {model_a_path} does not exist."

    responses_files = [model_a_path]

    if args.model_b:
        model_b_path = os.path.join(args.model_responses_dir, f'{args.model_b}_responses.jsonl')
        assert os.path.exists(model_b_path), f"Cannot load model responses: {model_b_path} does not exist."
        responses_files.append(model_b_path)

    responses_data = [[]] * len(responses_files)
    for i, path in enumerate(responses_files):
        with open(path, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                responses_data[i].append(line_data['response'])

    client = OpenAI()

    prefix = 'single' if len(responses_files) == 1 else 'pairwise'
    models = f'{args.model_a}_{args.model_b}' if len(responses_files) == 2 else args.model_a

    output_dir = os.path.join(args.model_responses_dir, prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{models}_judge_{args.model_judge}.jsonl')

    with open(output_file, "w") as fout:
        with (Progress(
                TextColumn(
                    f"Grading with - {args.model_judge} •" + "[progress.percentage]{task.percentage:>3.0f}%"
                ),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
        ) as p):
            for i, sample in p.track(enumerate(dataset), total=len(dataset)):
                if len(responses_files) == 1:
                    # single-answer grading
                    responses = responses_data[0]
                    prompt = single_grading_templates[sample['preference']].format(
                        instruction=sample['instruction'],
                        reference=sample['reference'],
                        response=responses[i]
                    )
                    rating = 'N/A'
                    for _ in range(5):
                        try:
                            response = client.chat.completions.create(**{
                                'model': args.model_judge,
                                'messages': [
                                    {'role': 'system', 'content': single_grading_system_prompt},
                                    {'role': 'user', 'content': prompt}
                                ],
                                'temperature': 0,
                                'max_tokens': 1024,
                            })
                            response = response.choices[0].message.content
                            match = re.search(r'\bRating:\s*(\d+)\b', response)
                            if match:
                                rating = int(match.group(1))
                                break
                        except openai.BadRequestError:
                            # most likely hallucination in the response
                            response = 'N/A'
                            logging.error(f'Could not process following response (idx={i}):\n\n{responses[i]}\n')
                    if rating == 'N/A':
                        logging.error(f'Could not process following response (idx={i}):\n\n{responses[i]}\n')
                    logger.info(f'Rating: {rating}')
                    json.dump({'judgement': response, 'rating': rating}, fout)
                    fout.write('\n')
                else:
                    # pairwise grading with responses position swapping
                    pass
            """
            for response, sample in p.track(zip(responses, dataset), total=len(dataset)):
                ratings = []
                # swap positions
                for r1, r2 in [(response, sample['gpt4_response']), (sample['gpt4_response'], response)]:
                    prompt = pairwise_grading_templates[sample['preference']].format(
                        instruction=sample['instruction'],
                        response_1=r1,
                        response_2=r2
                    )

                    try:
                        response = client.chat.completions.create(**{
                            'model': 'gpt-3.5-turbo',
                            'messages': [
                                {'role': 'system', 'content': pairwise_grading_system_prompt},
                                {'role': 'user', 'content': prompt}
                            ],
                            'temperature': 0,
                            'max_tokens': 1024,
                        })
                        response = response.choices[0].message.content
                    except openai.BadRequestError:
                        # most likely hallucination in the response
                        response = 'N/A'

                    ratings.append(response)
                f.write(' | '.join(ratings))
                f.write("\n")
            """


if __name__ == "__main__":
    main()
