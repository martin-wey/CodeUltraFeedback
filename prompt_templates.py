# Judgement prompts

# taken from UltraFeedback
# https://github.com/OpenBMB/UltraFeedback/blob/main/src/data_annotation/preference_templates.py#L1C17-L5C77
gpt4_judge_system_prompt = """Your role is to evaluate text quality based on given criteria.
You'll receive an instructional description ("Instruction") and four responses ("Response").
Understand and interpret instructions to evaluate effectively.
Provide annotations for each response with a rating and rationale.
The four responses given are independent, and should be evaluated separately."""


instruction_following_template = """# Instruction Following Assessment

Evaluate the assistant's fidelity to provided instructions. Assess how accurately the assistant's responses align with user directives, noting any deviations and their justification.

**Evaluation Criteria**:
- **Precision in Following Instructions**: Does the assistant adhere to the specifics of the provided instructions?
- **Justification for Deviations**: If deviations occur, are they justified by critical necessity or explicit user request?
- **Alignment with User Directives**: How well do the assistant's responses  match the user’s specified needs and expectations?
- **Necessity of Deviations**: Are any deviations from instructions made only in situations deemed absolutely necessary or upon direct user request?

**Scoring**: Rate outputs on a scale of 1 to 5:
1. **Non-Compliant**: The assistant frequently deviates from instructions without necessity or user consent.
2. **Minimally Compliant**: The assistant shows some adherence to instructions but deviates without strong justification.
3. **Moderately Compliant**: The assistant generally follows instructions, with deviations occurring but justified by necessity or user request.
4. **Highly Compliant**: The assistant closely follows instructions, with few deviations, all of which are well justified.
5. **Fully Compliant**: The assistant exhibits perfect adherence to instructions, with deviations only when critically necessary and explicitly approved by the user.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Responses:
<response 1> [Response 1]
<response 2> [Response 2]
<response 3> [Response 3]
<response 4> [Response 4]

### Output
#### Output for Response 1
Rating: [Rating for response 1]
Rationale: [Rationale for the rating in short sentences]

#### Output for Response 2
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 3
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 4
Rating: [Rating]
Rationale: [Rationale]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}
<response 3> {response_3}
<response 4> {response_4}

### Output
"""

code_readability_template = """# Code Readability Assessment

Evaluate the readability of code segments. Assess how comments and documentation contribute to understanding the code's logic, purpose, and operation.

**Evaluation Criteria**:
- **Clarity**: How clear and understandable are the code and its accompanying comments/documentation?
- **Conciseness**: Are the comments and documentation succinct yet informative?
- **Relevance**: Do the comments and documentation directly contribute to explaining the code's logic, objectives, and functionality?
- **Comprehensibility**: Can users of varying technical backgrounds easily grasp the code's purpose and how it works?

**Scoring**: Rate outputs on a scale of 1 to 5:
1. **Poor Readability**: The code is hard to follow, with little to no helpful comments/documentation.
2. **Basic Readability**: The code has minimal comments/documentation, offering limited clarity or insight.
3. **Good Readability**: The code is reasonably clear with comments/documentation that aid understanding, though some areas could be improved.
4. **Very Good Readability**: The code and comments/documentation are clear and concise, making the code's logic and purpose easily understandable.
5. **Excellent Readability**: The code exemplifies outstanding readability, with clear, concise, and comprehensive comments/documentation that make it accessible to all users.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Responses:
<response 1> [Response 1]
<response 2> [Response 2]
<response 3> [Response 3]
<response 4> [Response 4]

### Output
#### Output for Response 1
Rating: [Rating for response 1]
Rationale: [Rationale for the rating in short sentences]

#### Output for Response 2
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 3
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 4
Rating: [Rating]
Rationale: [Rationale]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}
<response 3> {response_3}
<response 4> {response_4}

### Output
"""

code_explanation_template = """# Code Explanation Quality Assessment

Evaluate the clarity and depth of explanations accompanying code segments. Assess how well the explanation helps in understanding the code's purpose, logic, and design choices.

**Evaluation Criteria**:
- **Clarity**: How easy is it to understand the explanation?
- **Depth**: Does the explanation cover the logic, structure, and decisions behind the code?
- **Relevance**: Is the explanation relevant to the code's purpose and design philosophy?
- **Accessibility**: Can a broad audience understand the explanation, regardless of their technical background?

**Scoring**: Rate outputs on a scale of 1 to 5:
1. **Inadequate**: Explanation is unclear, superficial, or missing.
2. **Basic**: Explanation covers fundamental points but lacks depth or clarity.
3. **Good**: Explanation is clear and somewhat detailed, but may miss some deeper insights.
4. **Very Good**: Explanation is clear, detailed, and covers most logic, structure, and decisions.
5. **Excellent**: Explanation is exceptionally clear, in-depth, and makes the code's purpose, design, and logic accessible to all users.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Responses:
<response 1> [Response 1]
<response 2> [Response 2]
<response 3> [Response 3]
<response 4> [Response 4]

### Output
#### Output for Response 1
Rating: [Rating for response 1]
Rationale: [Rationale for the rating in short sentences]

#### Output for Response 2
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 3
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 4
Rating: [Rating]
Rationale: [Rationale]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}
<response 3> {response_3}
<response 4> {response_4}

### Output
"""

code_complexity_template = """# Code Complexity and Efficiency Assessment

Evaluate the solutions and code provided by the assistant for their time efficiency and resource management. Assess how well the code optimizes computational time and resources while ensuring the accuracy and effectiveness of the implemented algorithms.

**Evaluation Criteria**:
- **Time Efficiency**: Does the code minimize computational time?
- **Resource Efficiency**: Does the code use resources (e.g., memory, CPU) judiciously?
- **Algorithm Effectiveness**: Are the chosen algorithms accurate and efficient in achieving the desired outcomes?
- **Optimization**: Has the code been optimized for quick processing without compromising the solution's correctness or efficiency?

**Scoring**: Rate outputs on a scale of 1 to 5:
1. **Inefficient**: The code is resource-heavy and slow, with little evidence of optimization for time efficiency.
2. **Somewhat Efficient**: The code shows some effort towards efficiency, but significant improvements are needed in time and resource management.
3. **Moderately Efficient**: The code balances time and resource use reasonably well, with effective algorithm selection, but there's room for optimization.
4. **Highly Efficient**: The code demonstrates strong time and resource efficiency, with well-chosen algorithms that provide swift and accurate results.
5. **Optimally Efficient**: The code exemplifies the best practices in time and resource efficiency, with optimal algorithm selection and execution that ensure maximum effectiveness with minimal resource expenditure.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Responses:
<response 1> [Response 1]
<response 2> [Response 2]
<response 3> [Response 3]
<response 4> [Response 4]

### Output
#### Output for Response 1
Rating: [Rating for response 1]
Rationale: [Rationale for the rating in short sentences]

#### Output for Response 2
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 3
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 4
Rating: [Rating]
Rationale: [Rationale]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}
<response 3> {response_3}
<response 4> {response_4}

### Output
"""

coding_style_template = """# Coding Style Assessment

Evaluate the coding style of provided code segments. Assess how well the code adheres to the best practices of the language, focusing on readability, maintainability, and efficiency in line with the language's idiomatic style.

**Evaluation Criteria**:
- **Readability**: Is the code easy to read and understand?
- **Maintainability**: Can the code be easily modified or extended?
- **Efficiency**: Does the code execute tasks in an efficient manner?
- **Adherence to Idiomatic Style**: Does the code follow the stylistic norms and conventions specific to the programming language?

**Scoring**: Rate outputs on a scale of 1 to 5:
1. **Non-Adherent**: The code largely ignores language conventions, resulting in poor readability and maintainability.
2. **Somewhat Adherent**: The code makes some effort to follow language conventions but lacks consistency in readability, maintainability, or efficiency.
3. **Moderately Adherent**: The code is generally in line with language conventions, offering fair readability and maintainability, with room for improvement in efficiency.
4. **Highly Adherent**: The code strongly adheres to language conventions, demonstrating good readability, maintainability, and efficiency.
5. **Exemplary Adherent**: The code exemplifies the best practices of the language, excelling in readability, maintainability, efficiency, and idiomatic style.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Responses:
<response 1> [Response 1]
<response 2> [Response 2]
<response 3> [Response 3]
<response 4> [Response 4]

### Output
#### Output for Response 1
Rating: [Rating for response 1]
Rationale: [Rationale for the rating in short sentences]

#### Output for Response 2
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 3
Rating: [Rating]
Rationale: [Rationale]

#### Output for Response 4
Rating: [Rating]
Rationale: [Rationale]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}
<response 3> {response_3}
<response 4> {response_4}

### Output
"""

# LLMs instruction prompts

system_messages = {
    "codellama": "You are an expert programmer. Your task is to solve the following instruction. You must wrap any code in your answer using ```.",
    "wizardcoder": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
    "deepseek-coder": "You are an expert programmer. Your task is to solve the following instruction. "
                      "You must wrap any code in your answer using ```.",
    "mistral-instruct-code": "You are an expert programmer. Your task is to solve the following instruction. "
                             "You must wrap any code in your answer using ```.",
    "wizardlm": "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    "llama2": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
              "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
              "Please ensure that your responses are socially unbiased and positive in nature.\n\n"
              "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. "
              "If you don’t know the answer to a question, please don’t share false information.\n"
}

principles = {
    "instruction-following": [
        "The assistant is required to strictly follow the provided instructions, making sure that all responses and actions are in exact accordance with user directives. Only in critical situations or upon direct user request should it deviate from these instructions.",
        "It's essential for the assistant to rigorously stick to the instructions given, aligning its responses and actions accurately with user demands. Deviations are permissible solely when indispensable or specifically asked for by the user.",
        "The assistant must ensure precise compliance with instructions, aligning its responses and actions directly with user requirements. Any deviation should occur only if absolutely necessary or if explicitly requested by the user.",
        "In following instructions, the assistant should adhere closely to what is asked, ensuring responses and actions match the user’s specified needs. Departure from these instructions is allowed only when critically needed or when the user explicitly allows it.",
        "The assistant is obliged to faithfully follow the instructions provided, ensuring its responses and actions mirror user expectations precisely. Only in necessary circumstances or at the user's express request should it deviate.",
        "The assistant needs to maintain strict adherence to user instructions, aligning its responses and actions specifically with the user’s requests. It should deviate from these instructions only when there is an absolute necessity or upon user’s explicit instruction.",
        "It is crucial for the assistant to be diligent in following user instructions, keeping its responses and actions in tight alignment with what is requested. Deviations should be considered only under essential conditions or if requested by the user.",
        "The assistant's responses and actions must conform closely to the instructions given by the user. It should only diverge from these instructions in situations of utmost necessity or when explicitly asked by the user.",
        "The assistant should execute tasks in strict accordance with user instructions, ensuring all actions and responses are directly in line with user requirements. Only in pressing situations or with clear user consent should it deviate.",
        "For the assistant, precise adherence to the given instructions is mandatory, with responses and actions reflecting user requests accurately. Deviating from these guidelines is acceptable only in critical instances or when the user specifically requests it."
    ],
    "readability": [
        "The assistant must ensure its code and responses are marked by ease of understanding, incorporating straightforward, brief comments and documentation that clarify the code's logic, purpose, and operational process, facilitating user comprehension and learning.",
        "It is crucial for the assistant to make its code and responses readily comprehensible, using clear and concise explanations in comments and documentation to illuminate the reasoning, objectives, and functions of the code, thus aiding in user education and application.",
        "The assistant should provide code and responses that are easily graspable, with lucid, succinct comments and documentation elucidating the underlying logic, goals, and functionality of the code, enhancing the user's ability to follow and absorb the material.",
        "In delivering code and responses, the assistant is expected to prioritize understandability, featuring clear, compact comments and documentation that detail the code’s logic, intent, and capabilities, thereby assisting users in their learning journey.",
        "The assistant is tasked with ensuring that both its code and responses are straightforward to comprehend, integrating clear, brief comments and documentation that explicate the code's reasoning, purpose, and mechanisms, thus streamlining the learning process for users.",
        "For effective readability, the assistant should ensure its code and responses are effortlessly intelligible, employing clear, concise comments and documentation that convey the logic, objectives, and functionality of the code, simplifying the learning curve for users.",
        "The assistant's code and responses need to be accessible in terms of understanding, necessitating the use of clear, brief comments and documentation that outline the logic, intent, and function of the code, thereby enhancing user comprehension and learning.",
        "The assistant has the responsibility to ensure its code and responses are straightforward to understand, necessitating lucid, concise comments and documentation that outline the code’s reasoning, purpose, and operational aspects, key to ensuring users can easily follow and benefit from the provided material.",
        "The assistant should maintain a high level of readability in both its code and responses, achieved by integrating clear, straightforward comments and documentation that illuminate the code’s reasoning, purpose, and functionality, to assist users in easily grasping and learning from the content.",
        "In providing code and responses, the assistant must focus on making them easily understandable, entailing the inclusion of lucid, succinct comments and documentation that explicate the code’s logic, aim, and functional aspects, essential for ensuring users can easily follow and benefit from the provided content."
    ],
    "complexity": [
        "The assistant is tasked with delivering solutions and code that prioritize time efficiency, focusing on reducing computational duration and resource expenditure while ensuring the accuracy and effectiveness of the algorithms used.",
        "It's essential for the assistant to offer solutions and code that are optimized for quick processing, designed to minimize the time and resources required for execution, while maintaining correctness and efficiency.",
        "The assistant must ensure that its solutions and code are developed with optimal time efficiency in mind, aiming to lessen computational time and resource use, and selecting algorithms that are both accurate and swift in execution.",
        "In providing solutions, the assistant should focus on achieving the highest time efficiency, minimizing the amount of computational time and resources needed, while ensuring the selected algorithms are both correct and efficient.",
        "The assistant's goal should be to present solutions and code that excel in time efficiency, designed to curtail the duration of computation and resource usage, while adhering to algorithms that are effective and precise.",
        "For every solution and piece of code, the assistant is expected to prioritize time efficiency, striving to reduce computational time and resources, and favoring methods that are not only accurate but also swift.",
        "The assistant needs to concentrate on offering solutions and code that are time-efficient, minimizing computational and resource demands, while choosing algorithms that are both effective and efficient.",
        "In its solutions and code, the assistant should aim for optimal time efficiency, designing methods that cut down on computational time and resource use, while ensuring the approaches used are both correct and swift.",
        "The assistant is required to prioritize time efficiency in its solutions and code, focusing on minimizing the time and resources involved in computation, and selecting algorithms that are not only accurate but also quick.",
        "The assistant's objective should be to provide solutions and code that excel in time efficiency, aiming to reduce both computational time and resource consumption, while utilizing algorithms that are effective and efficient in their operation."
    ],
    "style": [
        "The assistant should write code that exemplifies the best practices of the language used, focusing on readability, maintainability, and efficiency in line with the language's idiomatic style.",
        "It is important for the assistant to adopt a coding style that is characteristic of the language, balancing clarity, conciseness, and efficiency to produce easily understandable and efficient code.",
        "The assistant's coding approach should reflect the standard conventions of the language, prioritizing a style that is both effective and in harmony with the language’s best practices.",
        "In coding, the assistant must aim for a style that is not only syntactically correct but also adheres to the idiomatic nuances of the programming language, ensuring efficiency and readability.",
        "The assistant is expected to craft code that follows the established stylistic norms of the programming language, optimizing for a balance between clarity, efficiency, and language-specific best practices.",
        "The assistant should maintain a coding style that aligns with the language's conventional practices, ensuring that the code is efficient, readable, and easily maintained.",
        "Coding in a style that is representative of the language's community standards, the assistant should strive for code that is succinct, efficient, and easily interpreted by others.",
        "The assistant's coding should demonstrate a clear adherence to the stylistic principles of the language, focusing on creating code that is both high-performing and elegantly written.",
        "In its coding practices, the assistant should embody the ideal balance of efficiency, readability, and adherence to the language's stylistic conventions, ensuring optimal code performance.",
        "The assistant's coding style should be a reflection of the language’s standard practices, emphasizing a clean, efficient, and idiomatic approach to ensure both performance and readability."
    ],
    "explanation": [
        "The assistant is expected to provide detailed and accessible explanations for its code, clarifying the reasoning, architecture, and decisions involved, thereby aiding users in comprehending both the function and the rationale behind the code.",
        "It's important for the assistant to deliver comprehensive and intelligible explanations for the code, breaking down the logic, framework, and choices, to help users grasp the code's purpose and construction.",
        "The assistant must ensure that its code comes with clear, in-depth explanations, illuminating the underlying logic, structure, and choices, enabling users to not only understand the code's workings but also its design philosophy.",
        "In providing code, the assistant should also offer exhaustive and lucid explanations, elucidating the thought process, organization, and decisions behind the code, thus facilitating user understanding and application.",
        "The assistant should accompany its code with explicit and detailed explanations, shedding light on the logic, configuration, and decisions, helping users to comprehend and utilize the code effectively.",
        "For every piece of code, the assistant must provide clear, comprehensive explanations, detailing the logic, structure, and reasoning, enabling users to fully understand and apply the code in their contexts.",
        "The assistant is obliged to offer clear-cut and extensive explanations with its code, dissecting the logic, structure, and strategic choices, ensuring users gain a complete understanding of the code's purpose and design.",
        "Alongside its code, the assistant should present thorough and straightforward explanations, clarifying the underlying logic, framework, and choices, aiding users in understanding and adapting the code efficiently.",
        "The assistant must ensure that each code segment is accompanied by transparent and thorough explanations, unraveling the logic, structure, and choices, thus empowering users to grasp and leverage the code for their needs.",
        "It is essential for the assistant to couple its code with lucid, detailed explanations, unfolding the logic, composition, and decisions within the code, to help users not only understand its functionality but also the rationale behind its creation."
    ]
}

templates = {
    'codellama-34b-instruct': "[INST] {system_message}\n"
                              "{principle}\n\n"
                              "### Instruction: {instruction}\n"
                              "[/INST]",
    'codellama-13b-instruct': "[INST] {system_message}\n"
                              "{principle}\n\n"
                              "### Instruction: {instruction}\n"
                              "[/INST]",
    'codellama-7b-instruct': "[INST] {system_message}\n{principle}\n\n"
                             "### Instruction: {instruction}\n"
                             "[/INST]",
    'wizardcoder-33b': "{system_message}\n"
                       "{principle}\n\n"
                       "### Instruction:\n"
                       "{instruction}\n\n"
                       "### Response:",
    'wizardcoder-15b': "{system_message}\n"
                       "{principle}\n\n"
                       "### Instruction:\n"
                       "{instruction}\n\n"
                       "### Response:",
    'deepseek-coder-33b-instruct': "{system_message}\n"
                                   "{principle}\n\n"
                                   "### Instruction:\n"
                                   "{instruction}\n\n"
                                   "### Response:",
    'deepseek-coder-6.7b-instruct': "{system_message}\n"
                                    "{principle}\n\n"
                                    "### Instruction:\n"
                                    "{instruction}\n\n"
                                    "### Response:",
    'mistral-7b-instruct': "<|im_start|>system\n"
                           "{system_message} {principle}<|im_end|>\n"
                           "<|im_start|>user\n"
                           "{instruction}<|im_end|>\n"
                           "<|im_start|>assistant",
    'wizardlm-33b': "{system_message} {principle}\n\n"
                    "USER: {instruction}\n"
                    "ASSISTANT:",
    'wizardlm-7b': "{system_message} {principle}\n\n"
                   "USER: {instruction}\n"
                   "ASSISTANT:",
    'llama-2-13b-chat': "<s>[INST] <<SYS>>\n"
                        "{system_message}\n"
                        "{principle}\n"
                        "<</SYS>>\n"
                        "{instruction}[/INST]",
    'llama-2-70b-chat': "<s>[INST] <<SYS>>\n"
                        "{system_message}\n"
                        "{principle}\n"
                        "<</SYS>>\n"
                        "{instruction}[/INST]",
}

system_mappings = {
    'codellama-34b-instruct': "codellama",
    'codellama-13b-instruct': "codellama",
    'codellama-7b-instruct': "codellama",
    'wizardcoder-33b': "wizardcoder",
    'wizardcoder-15b': "wizardcoder",
    'deepseek-coder-33b-instruct': "deepseek-coder",
    'deepseek-coder-6.7b-instruct': "deepseek-coder",
    'mistral-7b-instruct': "mistral-instruct-code",
    'wizardlm-33b': "wizardlm",
    'wizardlm-7b': "wizardlm",
    'llama-2-13b-chat': "llama2",
    'llama-2-70b-chat': "llama2"
}

if __name__ == "__main__":
    template = templates["deepseek-coder-6.7b-instruct"]
    prompt_data = {
        'system_message': system_messages["deepseek-coder"],
        'principle': principles["instruction-following"][3],
        'instruction': "Write a function that computes fibonacci sequence."
    }
    print(template.format(**prompt_data))
