single_grading_system_prompt = "You are a helpful assistant."

instruction_following_template = """# Instruction Following Assessment

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user instruction displayed below. 
You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer according to the evaluation criteria described below. 
Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10.

Evaluate the assistant's fidelity to provided instructions. Assess how accurately the assistant's responses align with user directives, noting any deviations and their justification.

**Evaluation Criteria**:
- **Precision in Following Instructions**: Does the assistant adhere to the specifics of the provided instructions?
- **Justification for Deviations**: If deviations occur, are they justified by critical necessity or explicit user request?
- **Alignment with User Directives**: How well do the assistant's responses  match the user’s specified needs and expectations?
- **Necessity of Deviations**: Are any deviations from instructions made only in situations deemed absolutely necessary or upon direct user request?

## Format:

### Input
Instruction: [Clearly specify the instruction]

Reference Answer:
[Reference]

Assistant's Answer:
[Response]

### Output
Explanation: [Your explanation]
Rating: [Rating for response]

---

## Annotation

### Input
Instruction: {instruction}

Reference Answer:
{reference}

Assistant's Answer:
{response}

### Output
"""


code_readability_template = """# Code Readability Assessment

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user instruction displayed below. 
You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer according to the evaluation criteria described below. 
Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10.

Evaluate the readability of code segments. Assess how comments and documentation contribute to understanding the code's logic, purpose, and operation.

**Evaluation Criteria**:
- **Clarity**: How clear and understandable are the code and its accompanying comments/documentation?
- **Conciseness**: Are the comments and documentation succinct yet informative?
- **Relevance**: Do the comments and documentation directly contribute to explaining the code's logic, objectives, and functionality?
- **Comprehensibility**: Can users of varying technical backgrounds easily grasp the code's purpose and how it works?

## Format:

### Input
Instruction: [Clearly specify the instruction]

Reference Answer:
[Reference]

Assistant's Answer:
[Response]

### Output
Explanation: [Your explanation]
Rating: [Rating for response]

---

## Annotation

### Input
Instruction: {instruction}

Reference Answer:
{reference}

Assistant's Answer:
{response}

### Output
"""


code_explanation_template = """# Code Explanation Quality Assessment

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user instruction displayed below. 
You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer according to the evaluation criteria described below. 
Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10.

Evaluate the clarity and depth of explanations accompanying code segments. Assess how well the explanation helps in understanding the code's purpose, logic, and design choices.

**Evaluation Criteria**:
- **Clarity**: How easy is it to understand the explanation?
- **Depth**: Does the explanation cover the logic, structure, and decisions behind the code?
- **Relevance**: Is the explanation relevant to the code's purpose and design philosophy?
- **Accessibility**: Can a broad audience understand the explanation, regardless of their technical background?

## Format:

### Input
Instruction: [Clearly specify the instruction]

Reference Answer:
[Reference]

Assistant's Answer:
[Response]

### Output
Explanation: [Your explanation]
Rating: [Rating for response]

---

## Annotation

### Input
Instruction: {instruction}

Reference Answer:
{reference}

Assistant's Answer:
{response}

### Output
"""


code_complexity_template = """# Code Complexity and Efficiency Assessment

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user instruction displayed below. 
You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer according to the evaluation criteria described below. 
Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10.

Evaluate the solutions and code provided by the assistant for their time efficiency and resource management. Assess how well the code optimizes computational time and resources while ensuring the accuracy and effectiveness of the implemented algorithms.

**Evaluation Criteria**:
- **Time Efficiency**: Does the code minimize computational time?
- **Resource Efficiency**: Does the code use resources (e.g., memory, CPU) judiciously?
- **Algorithm Effectiveness**: Are the chosen algorithms accurate and efficient in achieving the desired outcomes?
- **Optimization**: Has the code been optimized for quick processing without compromising the solution's correctness or efficiency?

## Format:

### Input
Instruction: [Clearly specify the instruction]

Reference Answer:
[Reference]

Assistant's Answer:
[Response]

### Output
Explanation: [Your explanation]
Rating: [Rating for response]

---

## Annotation

### Input
Instruction: {instruction}

Reference Answer:
{reference}

Assistant's Answer:
{response}

### Output
"""

coding_style_template = """# Coding Style Assessment

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user instruction displayed below. 
You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer according to the evaluation criteria described below. 
Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10.

Evaluate the coding style of provided code segments. Assess how well the code adheres to the best practices of the language, focusing on readability, maintainability, and efficiency in line with the language's idiomatic style.

**Evaluation Criteria**:
- **Readability**: Is the code easy to read and understand?
- **Maintainability**: Can the code be easily modified or extended?
- **Efficiency**: Does the code execute tasks in an efficient manner?
- **Adherence to Idiomatic Style**: Does the code follow the stylistic norms and conventions specific to the programming language?

## Format:

### Input
Instruction: [Clearly specify the instruction]

Reference Answer:
[Reference]

Assistant's Answer:
[Response]

### Output
Explanation: [Your explanation]
Rating: [Rating for response]

---

## Annotation

### Input
Instruction: {instruction}

Reference Answer:
{reference}

Assistant's Answer:
{response}

### Output
"""

single_grading_templates = {
    'instruction-following': instruction_following_template,
    'readability': code_readability_template,
    'complexity': code_complexity_template,
    'style': coding_style_template,
    'explanation': code_explanation_template
}
