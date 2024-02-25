# `CodeUltraFeedback`
<div align="center">

### **Aligning large language models to coding preferences**

</div>

<p align="center">
    <a href="https://evalplus.github.io/leaderboard.html"><img src="https://img.shields.io/badge/%F0%9F%8F%86-Leaderboard-8A2BE2"></a>
    <a href="https://openreview.net/forum?id=1qvx610Cu7"><img src="https://img.shields.io/badge/Paper-ICSE'25-a55fed.svg"></a>
    <a href="https://huggingface.co/evalplus/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-CodeUltraFeedback-%23ff8811.svg"></a>
    <a href="https://github.com/martin-wey/CodeUltraFeedback/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/evalplus"></a>
</p>

<p align="center">
    <a href="#">🤔 About</a> •
    <a href="#">🧠 Models</a> •
    <a href="#">🤗 Datasets</a> •
    <a href="#">🏆 Leaderboard</a> •
    <a href="#">📝 Citation</a> •
    <a href="#">🙏 Acknowledgements</a>
</p>

> [!IMPORTANT]
> test readme

**Contact:** [Martin Weyssow](https://martin-wey.github.io/).

## About

<div style="text-align: center;">

![Overview of CodeUltraFeedback](assets/CodeUltraFeedback.svg)

</div>

## 🧠 Models

| Model                                     | Checkpoint                                                         |  Size   |     HumanEval (+)     |       MBPP (+)        | License                                                                            |
|:------------------------------------------|:-------------------------------------------------------------------|:-------:|:---------------------:|:---------------------:|:-----------------------------------------------------------------------------------|
| **CodeLlama-7B-Instruct-SFT**             | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-CL-7B)      |  `7B`   |      60.4 (55.5)      |      64.2 (52.6)      | [Llama2](https://ai.meta.com/llama/license/)                                       |
| **CodeLlama-7B-Instruct-DPO**             | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-S-CL-7B)    |  `7B`   |      70.7 (66.5)      |      68.4 (56.6)      | [Llama2](https://ai.meta.com/llama/license/)                                       |
| **CodeLlama-7B-Instruct-SFT+DPO**         | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-DS-6.7B)    |  `7B`   |      66.5 (60.4)      |      75.4 (61.9)      | [Llama2](https://ai.meta.com/llama/license/)                                       |
| **DeepSeek-Coder-6.7B-Instruct-SFT**      | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B)  | `6.7B`  |  **76.8** (**70.7**)  |  **75.7** (**64.4**)  | [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL)  |
| **DeepSeek-Coder-6.7B-Instruct-DPO**      | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B)  | `6.7B`  |  **76.8** (**70.7**)  |  **75.7** (**64.4**)  | [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL)  |
| **DeepSeek-Coder-6.7B-Instruct-SFT+DPO**  | 🤗 [HF Link](https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B)  | `6.7B`  |  **76.8** (**70.7**)  |  **75.7** (**64.4**)  | [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder/blob/main/LICENSE-MODEL)  |


## 🤗 Datasets
- **CodeUltraFeedback**
- **CodeUltraFeedback-SFT**
- **CodeUltraFeedback-Binarized**