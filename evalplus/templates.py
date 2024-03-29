sft_dpo_models_template = (
    "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' "
    "+ message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ "
    "'<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == "
    "'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% "
    "endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>\n```python' }}\n{% "
    "endif %}\n{% endfor %}"
)

codellama_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '\n' + message['content'] + '[/INST]' }}\n{% elif message['role'] == 'system' %}\n{{ '[INST] ' + message['content'] }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '```python' }}\n{% endif %}\n{% endfor %}"
