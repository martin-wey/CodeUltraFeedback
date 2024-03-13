# `CodeUltraFeedback`
<div align="center">

### **Aligning Large Language Models to Coding Preferences**

</div>

<p align="center">
    <a href="https://openreview.net/forum?id=1qvx610Cu7"><img src="https://img.shields.io/badge/ArXiV-ICSE'25-a55fed.svg"></a>
    <a href="https://huggingface.co/evalplus/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-CodeUltraFeedback-%23ff8811.svg"></a>
    <a href="https://github.com/martin-wey/CodeUltraFeedback/blob/main/LICENSE"><img src="https://img.shields.io/github/license/martin-wey/CodeUltraFeedback"></a>
</p>

<p align="center">
    <a href="#-about">🤔 About</a> •
    <a href="#-getting-started">🚀 Getting Started</a> •
    <a href="#-models">🧠 Models</a> •
    <a href="#-datasets-and-benchmark">🤗 Datasets</a> •
    <a href="#-citation">📝 Citation</a>
</p>

> [!NOTE]
> 
> [03-13-2024] 🏆 We are preparing a leaderboard for CODAL-Bench, stay tuned!
>
> [03-13-2024] 🔥 We release the first version of CodeUltraFeedback and CODAL-Bench.

**Contact:** If you have any inquiries or want to raise an issue, please feel free to contact [Martin Weyssow](https://martin-wey.github.io/), [martin.weyssow@umontreal.ca](mailto:martin.weyssow@umontreal.ca).


## About

<figure>
    <div align="center">
    <img src="assets/CodeUltraFeedback.svg"
         alt="CodeUltraFeedback Overview">
    <figcaption><i>Overview of CodeUltraFeedback dataset construction (see <a href="">Section II of our paper</a> for more details).</i></figcaption>
    </div>
</figure>


> Given the increasing coding capabilities of large language models (LLMs), the following question emerges:
> 
> _How well do these capabilities align with the expectations of developers, particularly concerning non-functional requirements such as code readability, efficiency, and adherence to best practices?_
>
> We believe existing benchmarks relying on automated metrics and static analysis tools are insufficient and too rigid for evaluating the broader capabilities of LLMs. 
> Instead, we believe LLM-as-a-judge offers a more viable alternative (_or proxy to human evaluation_) to evaluate LLMs while effectively considering the intricacies of natural and programming languages.

Our work features two main contributions: `CodeUltraFeedback` and `CODAL-Bench`, a dataset and benchmark for aligning LLMs to coding preferences and evaluating their alignment using LLM-as-a-judge.

`CodeUltraFeedback` is a preference dataset of complex coding instructions to align LLMs to coding preferences. 
It has an analogous construction procedure to [UltraFeedback](https://github.com/OpenBMB/UltraFeedback), featuring:

* ✨ **Complex instructions**: CodeUltraFeedback is based on a 10k subset of [MagiCoder Evol-Instruct](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) comprising open domain complex coding instructions.
* ✨ **Coding preferences**: CodeUltraFeedback includes 5 coding preferences, which are crucial to evaluate the broader capabilities of LLMs: **instruction-following**, **code explanation**, **code complexity and efficiency**, **code readability**, and **coding style**.
* ✨ **Large pool of LLMs**: We use a large pool of 14 LLMs from 8 model families to generate responses to the 10k instructions to consider diverse writing and coding styles.
* ✨ **LLM-as-a-judge and AI feedback**: We use GPT-3.5 as a judge for evaluating LLM responses, which annotates each response with both numerical and textual feedback. The AI feedback data can be leveraged for various applications, including model alignment through RLAIF, tuning a critic LLM, and more.

`CODAL-Bench` is a benchmark of 500 coding problems (_100 per coding preference_). We use LLM-as-a-judge with reference-guided single-answer grading using GPT-3.5 or GPT-4 to evaluate LLM alignment. 
The approach enables the judge LLM to provide consistent ratings and evaluate each LLM individually (similar to [MT-Bench](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge)). 

The following figure gives a broad overview of CodeUltraFeedback's construction procedure (see **Section II of our paper** for more details).

## 🚀 Getting Started 

We provide all the source code implemented to build CodeUltraFeedback and evaluate LLMs on CODAL-Bench.

> [!NOTE]
> 
> We are currently working on instructions to:
> 1. Build CodeUltraFeedback
> 2. Tune your own SFT and DPO LLMs
> 3. Evaluate LLMs on CODAL-Bench

## Models

| Model                             | Checkpoint                                                                      | Size | CODAL-Bench GPT-3.5<br/>(G-3.5, G-4) | CODAL-Bench GPT-4 <br/>(G-4) | HumanEval+<br/>(k=1, k=10) | License                                      |
|:----------------------------------|:--------------------------------------------------------------------------------|:----:|:------------------------------------:|:----------------------------:|:--------------------------:|:---------------------------------------------|
| **CodeLlama-7B-Instruct**         | 🤗 [HF Link](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)         | `7B` |             6.00 / 5.46              |             4.72             |        37.9 / 60.4         | [Llama2](https://ai.meta.com/llama/license/) |
| **CodeLlama-7B-Instruct-SFT**     | 🤗 [HF Link](https://huggingface.co/coseal/CodeLlama-7B-Instruct-sft-qlora)     | `7B` |             6.51 / 5.83              |             5.84             |    **51.2** / **82.9**     | [Llama2](https://ai.meta.com/llama/license/) |
| **CodeLlama-7B-Instruct-DPO**     | 🤗 [HF Link](https://huggingface.co/coseal/CodeLlama-7B-Instruct-dpo-qlora)     | `7B` |             7.15 / 6.79              |             5.08             |        42.3 / 80.5         | [Llama2](https://ai.meta.com/llama/license/) |
| **CodeLlama-7B-Instruct-SFT+DPO** | 🤗 [HF Link](https://huggingface.co/coseal/CodeLlama-7B-Instruct-sft-dpo-qlora) | `7B` |         **7.36** / **7.08**          |           **5.85**           |        43.1 / 75.6         | [Llama2](https://ai.meta.com/llama/license/) |

##  Datasets and Benchmark
- 🤗 **CodeUltraFeedback**: [https://huggingface.co/datasets/coseal/CodeUltraFeedback](https://huggingface.co/datasets/coseal/CodeUltraFeedback)
- 🤗 **Magicoder-Evol-Instruct-110K-sft**: [https://huggingface.co/datasets/coseal/Magicoder-Evol-Instruct-110K-sft](https://huggingface.co/datasets/coseal/Magicoder-Evol-Instruct-110K-sft)
- 🤗 **CodeUltraFeedback binarized**: [https://huggingface.co/datasets/coseal/CodeUltraFeedback_binarized](https://huggingface.co/datasets/coseal/CodeUltraFeedback_binarized)
- 🤗 **CODAL-Bench**: [https://huggingface.co/datasets/coseal/codal-bench](https://huggingface.co/datasets/coseal/codal-bench)

## 📝 Citation

