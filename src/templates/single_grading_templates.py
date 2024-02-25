single_grading_system_prompt = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user instruction displayed below.
You will be given a Reference Response and the Assistant's Response. Begin your evaluation by comparing the Assistant's Response with the Reference Response. 
Your overall evaluation needs to be reflective of the specified Evaluation Criteria. Be as objective as possible. 
After providing your rationale, you must rate the Assistant's Response on a scale of 1 to 10.
DO NOT give a response to the instruction, ONLY provide your rationale following by the rating.
"""


instruction_following_template = """# Instruction Following Assessment

Evaluation Criteria:
- Precision in Following Instructions: Does the assistant adhere to the specifics of the provided instructions?
- Justification for Deviations: If deviations occur, are they justified by critical necessity or explicit user request?
- Alignment with User Directives: How well do the assistant's responses  match the user’s specified needs and expectations?
- Necessity of Deviations: Are any deviations from instructions made only in situations deemed absolutely necessary or upon direct user request?

## Format:

### Instruction
[Clearly specify the instruction]

### Reference Response:
[Reference]

### Assistant's Response:
[Response]

### Output
Rationale: [Rationale for the rating in short sentences]
Rating: [Rating for Assistant's Response]

---

## Example of output:
Rationale: The assistant's response ...
Rating: 2/10

---

## Annotation

### Instruction
{instruction}

### Reference Response:
{reference}

### Assistant's Response:
{response}

### Output
"""


code_readability_template = """# Code Readability Assessment

Evaluation Criteria:
- Clarity: How clear and understandable are the code and its accompanying comments/documentation?
- Conciseness: Are the comments and documentation succinct yet informative?
- Relevance: Do the comments and documentation directly contribute to explaining the code's logic, objectives, and functionality?
- Comprehensibility: Can users of varying technical backgrounds easily grasp the code's purpose and how it works?

## Format:

### Instruction
[Clearly specify the instruction]

### Reference Response:
[Reference]

### Assistant's Response:
[Response]

### Output
Rationale: [Rationale for the rating in short sentences]
Rating: [Rating for Assistant's Response]

---

## Example of output:
Rationale: The assistant's response ...
Rating: 2/10

---

## Annotation

### Instruction
{instruction}

### Reference Response:
{reference}

### Assistant's Response:
{response}

### Output
"""


code_explanation_template = """# Code Explanation Quality Assessment

Evaluation Criteria:
- Clarity: How easy is it to understand the explanation?
- Depth: Does the explanation cover the logic, structure, and decisions behind the code?
- Relevance: Is the explanation relevant to the code's purpose and design philosophy?
- Accessibility: Can a broad audience understand the explanation, regardless of their technical background?

## Format:

### Instruction
[Clearly specify the instruction]

### Reference Response:
[Reference]

### Assistant's Response:
[Response]

### Output
Rationale: [Rationale for the rating in short sentences]
Rating: [Rating for Assistant's Response]

---

## Example of output:
Rationale: The assistant's response ...
Rating: 2/10

---

## Annotation

### Instruction
{instruction}

### Reference Response:
{reference}

### Assistant's Response:
{response}

### Output
"""


code_complexity_template = """# Code Complexity and Efficiency Assessment

Evaluation Criteria:
- Time Efficiency: Does the code minimize computational time?
- Resource Efficiency: Does the code use resources (e.g., memory, CPU) judiciously?
- Algorithm Effectiveness: Are the chosen algorithms accurate and efficient in achieving the desired outcomes?
- Optimization: Has the code been optimized for quick processing without compromising the solution's correctness or efficiency?

## Format:

### Instruction
[Clearly specify the instruction]

### Reference Response:
[Reference]

### Assistant's Response:
[Response]

### Output
Rationale: [Rationale for the rating in short sentences]
Rating: [Rating for Assistant's Response]

---

## Example of output:
Rationale: The assistant's response ...
Rating: 2/10

---

## Annotation

### Instruction
{instruction}

### Reference Response:
{reference}

### Assistant's Response:
{response}

### Output
"""

coding_style_template = """# Coding Style Assessment

Evaluation Criteria:
- Readability: Is the code easy to read and understand?
- Maintainability: Can the code be easily modified or extended?
- Efficiency: Does the code execute tasks in an efficient manner?
- Adherence to Idiomatic Style: Does the code follow the stylistic norms and conventions specific to the programming language?

## Format:

### Instruction
[Clearly specify the instruction]

### Reference Response:
[Reference]

### Assistant's Response:
[Response]

### Output
Rationale: [Rationale for the rating in short sentences]
Rating: [Rating for Assistant's Response]

---

## Example of output:
Rationale: The assistant's response ...
Rating: 2/10

---

## Annotation

### Instruction
{instruction}

### Reference Response:
{reference}

### Assistant's Response:
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
