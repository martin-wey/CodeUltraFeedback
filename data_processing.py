import random
import re
import argparse
import json

from datasets import load_dataset, load_from_disk
from transformers import set_seed

model_pool = [
    'gpt-4', 'gpt-3.5-turbo',
    'codellama-34b-instruct', 'codellama-13b-instruct', 'codellama-7b-instruct',
    'wizardcoder-33b', 'wizardcoder-15b',
    'deepseek-coder-33b-instruct', 'deepseek-coder-6.7b-instruct',
    'mistral-7b-instruct',
    'wizardlm-33b', 'wizardlm-7b',
    'llama-2-13b-chat', 'llama-2-70b-chat'
]

preferences = [
    'instruction-following',
    'readability',
    'complexity',
    'style',
    'explanation'
]


def contains_chinese_like_characters(text):
    # regex for chinese-like characters
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(pattern.search(text))


def create_dataset(output_dir: str = './dataset'):
    """
    Extract a 10k instructions subset of MagiCoder EvolInstruct dataset.
    Associate one coding preference with each sample.
    """
    seed = 42
    set_seed(seed)

    dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")["train"]
    print(f'-- Initial dataset size: {len(dataset)}')
    dataset = dataset.filter(lambda ex: not contains_chinese_like_characters(ex["instruction"]))
    print(f'-- Size after filtering chinese characters: {len(dataset)}')
    print(f'-- Selecting random sub-dataset...')
    dataset = dataset.shuffle(seed=seed).select(range(10000))
    dataset = dataset.remove_columns(['response'])
    dataset = dataset.add_column('preference', [None] * len(dataset))
    dataset = dataset.map(lambda ex, i: {'preference': preferences[i % len(preferences)]}, with_indices=True)
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.add_column('models', [None] * len(dataset))
    dataset = dataset.map(lambda ex: {'models': random.sample(model_pool, 4)})
    print(f'-- Saving preprocessed dataset...')
    dataset.save_to_disk(output_dir)


def merge_responses(dataset_dir: str = './dataset', responses_dir: str = './dataset/model_gen'):
    """Merge LLMs responses with instruction dataset."""
    dataset = load_from_disk(dataset_dir)
    dataset = dataset.add_column('responses', [[]] * len(dataset))

    responses_dict = {model: {} for model in model_pool}
    for model in model_pool:
        responses_path = f'{responses_dir}/{model}.jsonl'
        with open(responses_path, 'r') as f:
            for line in f:
                response = json.loads(line)
                responses_dict[model][response['instruction']] = response['response']

    def add_response(example):
        for model in example['models']:
            example['responses'].append({
                'model': model,
                'response': responses_dict[model][example['instruction']]
            })
        return example

    dataset = dataset.map(add_response)
    dataset.save_to_disk(f'{dataset_dir}_responses')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, default='./dataset')
    parser.add_argument('-r', '--responses_dir', type=str, default='./dataset/model_gen')
    parser.add_argument('-c', '--create_dataset', action='store_true')
    parser.add_argument('-m', '--merge_responses', action='store_true')
    args = parser.parse_args()

    if args.create_dataset:
        create_dataset(args.dataset_dir)

    if args.merge_responses:
        merge_responses(args.dataset_dir, args.responses_dir)
