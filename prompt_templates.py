system_message = ("You are an expert programmer. Your task is to generate a code that solves an instruction given in a "
                  "natural, human-like manner. You must provide a solution that run fast and has) "
                  "the lowest complexity in terms of time complexity, otherwise you will be penalized. "
                  "I’m going to tip $100 for a better solution! "
                  "Output your solution by strictly following this format:\n"
                  "```{language}\n"
                  "{code}\n"
                  "```")

codellama_instruct_template = "[INST] {system_message}\n### Instruction:\n{instruction}\n[/INST]"

templates = {
    'codellama_instruct': codellama_instruct_template
}


if __name__ == "__main__":
    prompt_data = {
        'system_message': system_message,
        'instruction': 'test'
    }
    prompt = templates['codellama_instruct'].format(**prompt_data)
    print(prompt)
