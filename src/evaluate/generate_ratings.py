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
    single_grading_templates
)
from src.utils import (
    H4ArgumentParser, ModelArguments, DataArguments, GenerationConfig,
    setup_logger, load_dataset, load_model_and_tokenizer
)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, GenerationConfig))
    model_args, data_args, generation_args = parser.parse()
    set_seed(42)

    logger = setup_logger(name="Evaluate - Ratings Generation")

    logger.info(f"Loading dataset: {data_args.dataset_name_or_path}")
    dataset = load_dataset(data_args.dataset_name_or_path, data_args.dataset_split)

    if data_args.references_path:
        logger.info(f"Loading references: {data_args.references_path}")
        assert (os.path.exists(data_args.references_path),
                f"Cannot load references file: {data_args.references_path} does not exist.")
        with open(data_args.references_path, 'r') as f:
            references = [json.loads(l) for l in f]
        references_instructions = [r['instruction'] for r in references]

        def get_response(example):
            index = references_instructions.index(example['instruction'])
            return {'reference': references[index]['response']}

        dataset = dataset.add_column('reference', [None] * len(dataset))
        dataset = dataset.map(get_response)

    logger.info("Loading responses...")
    model_a_path = os.path.join(data_args.model_responses_dir, f'{model_args.model_a}_responses.jsonl')
    assert os.path.exists(model_a_path), f"Cannot load model responses: {model_a_path} does not exist."

    responses_files = [model_a_path]

    if model_args.model_b:
        model_b_path = os.path.join(data_args.model_responses_dir, f'{model_args.model_b}_responses.jsonl')
        assert os.path.exists(model_b_path), f"Cannot load model responses: {model_b_path} does not exist."
        responses_files.append(model_b_path)

    responses_data = [[] for _ in range(len(responses_files))]
    for i, path in enumerate(responses_files):
        with open(path, 'r') as f:
            for line in f:
                line_data = json.loads(line)
                responses_data[i].append(line_data['response'])

    logger.info(f"Loading model: {model_args.model_name}")
    generator, _, _ = load_model_and_tokenizer(model_args, generation_args)

    prefix = 'single' if len(responses_files) == 1 else 'pairwise'
    models = f'{model_args.model_a}_{model_args.model_b}' if len(responses_files) == 2 else model_args.model_a

    output_dir = os.path.join(data_args.model_responses_dir, prefix)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{models}_judge_{model_args.model_judge}.jsonl')

    with open(output_file, "w") as fout:
        progress_bar_prefix = "Single-answer grading" if len(responses_files) == 1 else "Pairwise grading"
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
                if len(responses_files) == 1:
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
                    try:
                        response = generator(
                            system_prompt=single_grading_system_prompt,
                            user_prompt=prompt
                        )
                        match = re.search(r'\bRating:\s*(\d+)\b', response)
                        if match:
                            rating = int(match.group(1))
                    except Exception as e:
                        logger.error(e)
                        rating = "N/A"

                    logger.info(f'Rating: {rating}')
                    json.dump({'judgement': response, 'rating': rating}, fout)
                    fout.write('\n')
                """
                else:
                    # pairwise grading with responses position swapping.
                    #
                    # Note: this grading alternative is not included in CodeUltraFeedback paper.
                    #       although we tried position swapping to mitigate position bias in LLM judges,
                    #       we found the results to be too inconsistent.

                    response_a = responses_data[0][i]
                    response_b = responses_data[1][i]

                    winner_map = {"A": "model_1", "B": "model_2"}
                    results = {
                        "model_1": args.model_a,
                        "model_2": args.model_b
                    }

                    for j, (r_a, r_b) in enumerate([(response_a, response_b), (response_b, response_a)]):
                        prompt = pairwise_grading_templates[sample['preference']].format(
                            instruction=sample['instruction'],
                            response_a=r_a,
                            response_b=r_b
                        )
                        winner = "N/A"
                        for _ in range(5):
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
                                if "[[A]]" in response:
                                    winner = "A"
                                    break
                                elif "[[B]]" in response:
                                    winner = "B"
                                    break
                                elif "[[C]]" in response:
                                    winner = "tie"
                                    break
                            except openai.BadRequestError:
                                # most likely hallucination in the response
                                winner = "N/A"
                        model_winner = winner_map.get(winner, winner)
                        logger.info(f'Winner: {model_winner}')
                        results[f"round{j+1}_winner"] = model_winner
                        results[f"round{j+1}_response"] = response
                        winner_map = {"A": "model_2", "B": "model_1"}
                    json.dump(results, fout)
                    fout.write("\n")
                """


if __name__ == "__main__":
    main()
