import random
import re

from datasets import load_dataset
from utils import preferences, set_seed, model_pool


def contains_chinese_like_characters(text):
    # regex for chinese-like characters
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    return bool(pattern.search(text))


if __name__ == "__main__":
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
    dataset.save_to_disk("./dataset")
