import logging
import random

import torch
import transformers
from datasets import load_from_disk
from peft import LoraConfig
from rich.logging import RichHandler
from transformers import set_seed, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
from unsloth import FastLanguageModel

from alignment import (
    DataArguments,
    SFTConfig,
    H4ArgumentParser,
    ModelArguments,
    split_dataset,
    maybe_insert_system_message
)

logger = logging.getLogger("rich")


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )
    log_level = training_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    dataset = load_from_disk(data_args.dataset_dir)
    raw_datasets = split_dataset(dataset)
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #######################
    # Load pretrained model
    #######################
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=training_args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=model_args.lora_target_modules,
        bias="none",
        use_gradient_checkpointing=training_args.gradient_checkpointing,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if data_args.truncation_side is not None:
        tokenizer.truncation_side = data_args.truncation_side

    if data_args.chat_template is not None:
        tokenizer.chat_template = data_args.chat_template

    #####################
    # Apply chat template
    #####################
    def apply_chat_template(example):
        messages = [
            {'content': example['instruction'], 'role': 'user'},
            {'content': example['response'], 'role': 'assistant'}
        ]
        maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return example

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template"
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    ########################
    # Initialize the Trainer
    ########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
