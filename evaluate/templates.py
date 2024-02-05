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
The Python code must be between ```python and ``` tags.

Provide a self-contained Python script that solves the following problem:
```python
{prompt}
```

Here is a Python script that solves the problem:
```python
[/INST]
"""
    mbpp: str = """"""


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
