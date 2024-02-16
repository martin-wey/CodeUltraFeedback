'''
pairwise_grading_system_prompt = """Your role is to evaluate text quality based on given criteria.
You'll receive an instructional description ("Instruction") and two responses ("Responses").
Understand and interpret instructions to evaluate effectively.
Provide annotations for each response with a rating and rationale.
The two responses given are independent, and should be evaluated separately."""
'''

pairwise_grading_system_prompt = """Your role is to evaluate text quality based on given criteria.
You'll receive an instructional description ("Instruction") and two responses ("Responses").
Understand and interpret instructions to evaluate effectively.
Provide annotations for each response with a rating."""

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

You MUST ONLY output ratings of 1, 2, 3, 4 or 5.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Response:
Responses:
<response 1> [Response 1]
<response 2> [Response 2]

### Output
Ratings: [[Rating for Response 1], [Rating for Response 2]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}

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

You MUST ONLY output ratings of 1, 2, 3, 4 or 5.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Response:
Responses:
<response 1> [Response 1]
<response 2> [Response 2]

### Output
Ratings: [[Rating for Response 1], [Rating for Response 2]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}

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

You MUST ONLY output ratings of 1, 2, 3, 4 or 5.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Response:
Responses:
<response 1> [Response 1]
<response 2> [Response 2]

### Output
Ratings: [[Rating for Response 1], [Rating for Response 2]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}

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

You MUST ONLY output ratings of 1, 2, 3, 4 or 5.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Response:
Responses:
<response 1> [Response 1]
<response 2> [Response 2]

### Output
Ratings: [[Rating for Response 1], [Rating for Response 2]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}

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

You MUST ONLY output ratings of 1, 2, 3, 4 or 5.

## Format:

### Input
Instruction: [Clearly specify the instruction]

Response:
Responses:
<response 1> [Response 1]
<response 2> [Response 2]

### Output
Ratings: [[Rating for Response 1], [Rating for Response 2]

---

## Annotation

### Input
Instruction: {instruction}

Responses:
<response 1> {response_1}
<response 2> {response_2}

### Output
"""

pairwise_grading_templates = {
    'instruction-following': instruction_following_template,
    'readability': code_readability_template,
    'complexity': code_complexity_template,
    'style': coding_style_template,
    'explanation': code_explanation_template
}
