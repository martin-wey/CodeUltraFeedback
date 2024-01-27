system_message = ("You are an expert programmer. Your task is to generate a code that solves an instruction given in a "
                  "natural, human-like manner. You must provide a solution that run fast and has) "
                  "the lowest complexity in terms of time complexity, otherwise you will be penalized. "
                  "I’m going to tip $100 for a better solution! "
                  "Output your solution by strictly following this format:\n"
                  "```{language}\n"
                  "{code}\n"
                  "```")

templates = {
    'codellama-34b-instruct': "[INST] {system_message}\n### Instruction:\n{instruction}\n[/INST]",
    'codellama-13b-instruct': "[INST] {system_message}\n### Instruction:\n{instruction}\n[/INST]",
    'codellama-7b-instruct': "[INST] {system_message}\n### Instruction:\n{instruction}\n[/INST]",
    'wizardcoder-33b': "{system_message}\n\n### Instruction:\n{instruction}\n\n### Response:",
    'wizardcoder-15b': "{system_message}\n\n### Instruction:\n{instruction}\n\n### Response:",
    'deepseek-coder-33b-instruct': "{system_message}\n\n### Instruction:\n{instruction}\n\n### Response:",
    'deepseek-coder-6.7b-instruct': "{system_message}\n\n### Instruction:\n{instruction}\n\n### Response:",
    'mistral-7b-instruct-code': "<|im_start|>{system_message}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant",
    'wizardlm-33b': "{system_message}\n\nUSER: {instruction}\nASSISTANT:",
    'wizardlm-7b': "{system_message}\n\nUSER: {instruction}\nASSISTANT:",
    'llama-2-13b-chat': "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n{instruction}[/INST]",
    'llama-2-70b-chat': "[INST] <<SYS>>\n{system_message}\n<</SYS>>\n{instruction}[/INST]",
}


if __name__ == "__main__":
    prompt_data = {
        'system_message': system_message,
        'instruction': 'test'
    }
    prompt = templates['codellama_instruct'].format(**prompt_data)
    print(prompt)
