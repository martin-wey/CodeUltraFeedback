from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the model on HuggingFace Hub or local path."}
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the model, used for output files naming."}
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "Revision of the model on HuggingFace Hub or local path."}
    )

    # Arguments for CODAL-Bench
    model_test: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the model to judge when generating ratings."}
    )
    model_judge: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the LLM judge (must be an OpenAI model)."}
    )
    model_reference: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the model responses to use as references for single-answer grading."}
    )


@dataclass
class DataArguments:
    dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the dataset on HuggingFace Hub or local path."}
    )
    dataset_split: Optional[str] = field(
        default=None,
        metadata={"help": "Split of the dataset to load."}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory path."}
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template to use."}
    )

    # Arguments for CODAL-Bench
    model_responses_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory containing model responses."}
    )

    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class GenerationConfig:
    max_new_tokens: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of new tokens to generate."},
    )
    temperature: Optional[int] = field(
        default=0.8,
        metadata={"help": "Temperature value for generation."},
    )
    top_p: Optional[int] = field(
        default=0.8,
        metadata={"help": "Top-p value for generation."},
    )
    greedy: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use greedy decoding."}
    )
