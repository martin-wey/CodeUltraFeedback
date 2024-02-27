from dataclasses import dataclass


@dataclass
class CodeLlamaInstructTemplate:
    template: str = """[INST] You are an expert programmer. Your task is to solve the following instruction. You must wrap any code in your answer using ```.
{principle}

### Instruction: {instruction}
[/INST]
"""


@dataclass
class WizardCoderTemplate:
    template: str = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
{principle}

### Instruction:
{instruction}

### Response:
"""


@dataclass
class DeepSeekCoderInstructTemplate:
    template: str = """You are an expert programmer. Your task is to solve the following instruction. You must wrap any code in your answer using ```.
{principle}

### Instruction:
{instruction}

### Response:
"""


@dataclass
class MistralInstructTemplate:
    template: str = """<|im_start|>system
You are an expert programmer. Your task is to solve the following instruction. You must wrap any code in your answer using ```. {principle}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""


@dataclass
class WizardLMTemplate:
    template: str = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. {principle}

USER: {instruction}
ASSISTANT:
"""


@dataclass
class Llama2ChatTemplate:
    template: str = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don’t know the answer to a question, please don’t share false information.

{principle}
<</SYS>>
{instruction}[/INST]
"""


model_templates = {
    'codellama-34b-instruct': CodeLlamaInstructTemplate,
    'codellama-13b-instruct': CodeLlamaInstructTemplate,
    'codellama-7b-instruct': CodeLlamaInstructTemplate,
    'wizardcoder-33b': WizardCoderTemplate,
    'wizardcoder-15b': WizardCoderTemplate,
    'deepseek-coder-33b-instruct': DeepSeekCoderInstructTemplate,
    'deepseek-coder-6.7b-instruct': DeepSeekCoderInstructTemplate,
    'mistral-7b-instruct': MistralInstructTemplate,
    'wizardlm-33b': WizardLMTemplate,
    'wizardlm-7b': WizardLMTemplate,
    'llama-2-13b-chat': Llama2ChatTemplate,
    'llama-2-70b-chat': Llama2ChatTemplate
}
