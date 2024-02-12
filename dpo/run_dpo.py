import logging
import random

import transformers
from datasets import load_from_disk
from rich.logging import RichHandler
from transformers import set_seed
from trl import DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_tokenizer
)
from principles import principles

logger = logging.getLogger("rich")


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig))
    model_args, data_args, training_args = parser.parse()

    logging.basicConfig(
        level="NOTSET",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
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
    raw_datasets = load_from_disk(data_args.dataset_dir)
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    #####################
    # Apply chat template
    #####################
    def apply_chat_template(example):
        prompt_msg = [
            {'content': random.choice(principles[example['preference']]), 'role': 'system'},
            {'content': example['instruction'], 'role': 'user'},
        ]
        chosen_msg = [
            {'content': example['chosen'], 'role': 'assistant'}
        ]
        rejected_msg = [
            {'content': example['rejected'], 'role': 'assistant'}
        ]
        example['text_chosen'] = tokenizer.apply_chat_template(chosen_msg, tokenize=False)
        example['text_rejected'] = tokenizer.apply_chat_template(rejected_msg, tokenize=False)
        example['text_prompt'] = tokenizer.apply_chat_template(prompt_msg, tokenize=False)

        return example

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template"
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Loading pretrained model ***")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_args.model_name_or_path,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=model_args.load_in_4bit
    )

    PatchDPOTrainer()
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        beta=training_args.beta,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        tokenizer=tokenizer,
        max_length=training_args.max_length,
        max_prompt_length=training_args.max_prompt_length,
        loss_type=training_args.loss_type,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
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
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    logger.info("*** Training complete! ***")


if __name__ == '__main__':
    main()
