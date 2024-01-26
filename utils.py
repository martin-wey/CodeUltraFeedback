import random

import numpy as np
import torch
from datasets import load_dataset


def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    dataset = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")["train"]
    dataset = dataset.shuffle(seed=42).select(range(10000))
    dataset.save_to_disk("./dataset")
