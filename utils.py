import random

import numpy as np
import torch


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


def set_seed(seed):
    print("set seed:", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
