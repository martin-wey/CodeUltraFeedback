# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random

from datasets import DatasetDict, Dataset


def maybe_insert_system_message(messages, tokenizer):
    if messages[0]["role"] == "system":
        return

    # chat template can be one of two attributes, we check in order
    chat_template = tokenizer.chat_template
    if chat_template is None:
        chat_template = tokenizer.default_chat_template

    # confirm the jinja template refers to a system message before inserting
    if "system" in chat_template:
        messages.insert(0, {"role": "system", "content": ""})


def split_dataset(
    dataset: Dataset,
    train_size=.95,
    test_size=.05
):
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    num_samples = len(dataset)
    train_end = int(train_size * num_samples)
    test_end = train_end + int(test_size * num_samples)

    train_indices = indices[:train_end]
    test_indices = indices[train_end:test_end]

    train_data = dataset.select(train_indices)
    test_data = dataset.select(test_indices)

    return DatasetDict({
        'train': train_data,
        'test': test_data
    })
