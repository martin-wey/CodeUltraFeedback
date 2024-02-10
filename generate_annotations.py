import argparse
import json
import os
import random
import re
import time

from datasets import load_from_disk
from openai import OpenAI
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from templates import gpt_judge_system_prompt, judge_templates


# Adapted from https://github.com/OpenBMB/UltraFeedback/blob/main/src/data_annotation/annotate_preference.py#L18
def process(responses):
    # @todo: better handling of exceptions and errors in response formatting
    annotations = {'ratings': [], 'rationales': []}
    responses = responses.split("\n\n")
    print(responses)
    if len(responses) != 4:
        annotations['rationales'] = responses
        return annotations
    try:
        pattern = r"Rating: (.+?)\nRationale: (.+)"
        for response in responses:
            matches = re.search(pattern, response, re.DOTALL)
            try:
                annotations['ratings'].append(re.findall(r'\b\d+\b', matches.group(1))[0] if matches.group(1) not in ['N/A', '', ' '] else "N/A")
            except Exception:
                print(responses)
                raise Exception
            annotations['rationales'].append(matches.group(2))
    except ValueError or AttributeError as e:
        annotations['rationales'] = responses
        return annotations
    except IndexError as e:
        print(responses)
        raise IndexError
    return annotations


class APICaller:
    def __init__(self, model, client):
        self.model = model
        self.client = client

    def __call__(self, system_prompt, user_prompt):
        content = None
        for _ in range(1):
            try:
                response = self.client.chat.completions.create(**{
                    'model': self.model,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt}
                    ],
                    'temperature': 0,
                    'max_tokens': 500,
                    'top_p': 0.6,
                    'presence_penalty': 0,
                    'frequency_penalty': 0
                })
                content = response.choices[0].message.content
            except Exception as e:
                time.sleep(1)
            else:
                break
        return content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, default='./dataset_responses')
    parser.add_argument("-o", "--output_dir", type=str, default='./dataset_responses/annotations')
    parser.add_argument('-m', '--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument("-st", "--start", default=0, type=int)
    args = parser.parse_args()

    dataset = load_from_disk(args.dataset_dir)

    if args.start > 0:
        dataset = dataset.select(range(args.start, len(dataset)))
        print(dataset)

    client = OpenAI()
    generator = APICaller(args.model, client)

    if args.start > 0:
        file_name = f'annotations_{args.model}_{args.start}.jsonl'
    else:
        file_name = f'annotations_{args.model}.jsonl'

    with open(os.path.join(args.output_dir, file_name), "w") as f:
        with (Progress(
                TextColumn(
                    f"Annotations - {args.model} •" + "[progress.percentage]{task.percentage:>3.0f}%"
                ),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
        ) as p):
            for sample in p.track(dataset, total=len(dataset)):
                format_input = {'instruction': sample['instruction']}
                # random order to avoid positional bias.
                # ideally, we need to run multiple times with different shuffling.
                order = list(range(4))
                random.shuffle(order)
                format_input.update({f'response_{i+1}': sample['responses'][o]['response'] for i, o in enumerate(order)})

                response = generator(system_prompt=gpt_judge_system_prompt,
                                     user_prompt=judge_templates[sample['preference']].format(**format_input))
                if response is not None:
                    annotations = process(response)
                    annotations['order'] = order
                else:
                    annotations = {'ratings': [], 'rationales': [], 'order': order}
                json.dump(annotations, f)
                f.write("\n")
