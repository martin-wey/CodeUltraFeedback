import random
import re
import argparse
import json

from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
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
    dataset.save_to_disk('dataset_responses')


def merge_annotations(dataset_dir: str = './dataset_responses', annotations_file: str = None):
    dataset = load_from_disk(dataset_dir)
    dataset = dataset.add_column('annotations', [[]] * len(dataset))

    with open(annotations_file, 'r') as f:
        annotations = [json.loads(l) for l in f]

    def add_annotations(example, idx):
        for i, rating, rationale in zip(annotations[idx]['order'], annotations[idx]['ratings'],
                                        annotations[idx]['rationales']):
            example['annotations'].append({
                'model': example['responses'][i]['model'],
                'rating': rating,
                'rationale': rationale
            })
        return example

    dataset = dataset.map(add_annotations, with_indices=True)
    dataset.save_to_disk('dataset_annotations')


def split_dataset(dataset_dir: str = './dataset_annotations',
                  train_size: float = 0.9,
                  test_size: float = 0.1,
                  seed: int = 42):
    random.seed(seed)
    dataset = load_from_disk(dataset_dir)

    train_datasets = []
    test_datasets = []
    for pref in preferences:
        ds = dataset.filter(lambda ex: ex['preference'] == pref)

        indices = list(range(len(ds)))
        random.shuffle(indices)

        num_samples = len(ds)
        train_end = int(train_size * num_samples)
        test_end = train_end + int(test_size * num_samples)

        train_indices = indices[:train_end]
        test_indices = indices[train_end:test_end]

        train_data = ds.select(train_indices)
        test_data = ds.select(test_indices)

        train_datasets.append(train_data)
        test_datasets.append(test_data)

    train_dataset = concatenate_datasets(train_datasets)
    test_dataset = concatenate_datasets(test_datasets)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })

    dataset_dict.save_to_disk('dataset_splits')


def binarize_dataset(dataset_dir: str = None):
    dataset = load_from_disk(dataset_dir)

    def select_responses(example):
        ratings = {annotation['model']: int(annotation['rating']) for annotation in example['annotations']
                   if annotation['rating'] != 'N/A'}

        max_rating = max(ratings.values())
        highest_rated_models = [model for model, rating in ratings.items() if rating == max_rating]
        random_highest_rated_model = random.choice(highest_rated_models)

        min_rating = min(ratings.values())
        lowest_rated_models = [model for model, rating in ratings.items() if rating == min_rating]
        random_lowest_rated_model = random.choice(lowest_rated_models)

        chosen_model_response = list(filter(lambda ex: ex['model'] == random_highest_rated_model, example['responses']))[0]
        rejected_model_response = list(filter(lambda ex: ex['model'] == random_highest_rated_model, example['responses']))[0]

        example['chosen'] = chosen_model_response['response']
        example['rejected'] = rejected_model_response['response']
        example['rating_chosen'] = ratings[random_highest_rated_model]
        example['rating_rejected'] = ratings[random_lowest_rated_model]
        example['model_chosen'] = random_highest_rated_model
        example['model_rejected'] = random_lowest_rated_model

        return example

    dataset = dataset.map(select_responses)

    dataset.save_to_disk('dataset_binarized')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, default='./dataset')
    parser.add_argument('-r', '--responses_dir', type=str, default='./dataset/model_gen')
    parser.add_argument('-a', '--annotations_file', type=str, default='annotations_gpt-3.5-turbo.jsonl')
    parser.add_argument('-c', '--create_dataset', action='store_true')
    parser.add_argument('-mr', '--merge_responses', action='store_true')
    parser.add_argument('-ma', '--merge_annotations', action='store_true')
    parser.add_argument('-s', '--split_dataset', action='store_true')
    parser.add_argument('-b', '--binarize_dataset', action='store_true')
    args = parser.parse_args()

    if args.create_dataset:
        create_dataset(args.dataset_dir)

    if args.merge_responses:
        merge_responses(args.dataset_dir, args.responses_dir)

    if args.merge_annotations:
        merge_annotations(args.dataset_dir, args.annotations_file)

    if args.split_dataset:
        split_dataset(args.dataset_dir)

    if args.binarize_dataset:
        binarize_dataset(args.dataset_dir)