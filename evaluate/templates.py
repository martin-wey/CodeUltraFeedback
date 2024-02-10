from dataclasses import dataclass


@dataclass
class Template:
    humaneval: str
    mbpp: str

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class CodeLlamaInstructTemplate(Template):
    humaneval: str = """[INST] Your task is to write a Python function to solve a programming problem.
The Python code must be between [PYTHON] and [/PYTHON] tags.

Provide a self-contained Python script that solves the following problem:
[PYTHON]
{prompt}
[/PYTHON]
[/INST]
"""
    mbpp: str = """You are an expert Python programmer, and here is your task: {prompt}

"""


@dataclass
class DeepSeekCoderInstructTemplate(Template):
    humaneval: str = """"""
    mbpp: str = """"""


@dataclass
class WizardCoderTemplate(Template):
    humaneval: str = """"""
    mbpp: str = """"""


templates = {
    'codellama-7b-instruct': CodeLlamaInstructTemplate(),
    'codellama-13b-instruct': CodeLlamaInstructTemplate(),
    'codellama-34b-instruct': CodeLlamaInstructTemplate(),
}
