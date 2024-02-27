import logging
import time

import datasets
import openai
import torch
from auto_gptq import exllama_set_max_input_length
from datasets import load_from_disk
from openai import OpenAI
from rich.logging import RichHandler
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from unsloth import FastLanguageModel


def setup_logger(name="logger", level=logging.INFO, log_file=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicate messages
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create RichHandler for console logging
    rich_handler = RichHandler()
    rich_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(rich_handler)

    # Optionally add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

    return logger


def load_dataset(dataset_name_or_path, split=None):
    if "datasets/" in dataset_name_or_path:
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = datasets.load_dataset(dataset_name_or_path)

    if split is not None and split in ["train", "test"]:
        return dataset[split]
    return dataset


def load_model_and_tokenizer(model_args, generation_args):
    if not model_args.model_name_or_path and "gpt-" in model_args.model_name:
        client = OpenAI()
        generator = APICaller(
            model=model_args.model_name,
            client=client,
            temperature=generation_args.temperature,
            max_tokens=generation_args.max_new_tokens,
            top_p=generation_args.top_p
        )
        return generator, None, None
    elif "GPTQ" in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            device_map="auto",
            trust_remote_code=False,
            torch_dtype=torch.float16,
        )
        if 'actorder' in model_args.revision:
            model = exllama_set_max_input_length(model, 8192)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return generator, model, tokenizer
    elif "-sft" in model_args.model_name or "-dpo" in model_args.model_name:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_args.model_name_or_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True
        )
        FastLanguageModel.for_inference(model)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return generator, model, tokenizer

    raise Exception("Unknown model name {}".format(model_args.model_name))


class APICaller:
    def __init__(self, model, client=None, retries=5, temperature=0.8, max_tokens=1024, top_p=1.0):
        self.model = model
        self.client = client or OpenAI()
        self.retries = retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

    def __call__(self, system_prompt, user_prompt):
        attempt = 0
        error = ""
        while attempt < self.retries:
            try:
                response = self.client.chat.completions.create(**{
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p
                })
                return response.choices[0].message.content
            except openai.BadRequestError:
                error = ("An error occurred: OpenAI could not process the prompt. "
                         "This is most likely due to repetitive words in the prompt.")
                break
            except Exception:
                error = "An error occurred."
                attempt += 1
        return error
