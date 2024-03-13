# `CodeUltraFeedback`
<div align="center">

### **Aligning Large Language Models to Coding Preferences**

</div>

<p align="center">
    <a href="https://openreview.net/forum?id=1qvx610Cu7"><img src="https://img.shields.io/badge/ArXiV-ICSE'25-a55fed.svg"></a>
    <a href="https://huggingface.co/evalplus/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-CodeUltraFeedback-%23ff8811.svg"></a>
    <a href="https://github.com/martin-wey/CodeUltraFeedback/blob/main/LICENSE"><img src="https://img.shields.io/github/license/martin-wey/CodeUltraFeedback"></a>
"></a>
</p>

<p align="center">
    <a href="#">🤔 About</a> •
    <a href="#">🧠 Models</a> •
    <a href="#">🤗 Datasets</a> •
    <a href="#">📝 Citation</a> •
    <a href="#">🙏 Acknowledgements</a>
</p>

## News
> [!NOTE]
> 
> [03-13-2024] 🔥 We release the first version of CodeUltraFeedback and CODAL-Bench.

**Contact:** If you have any inquiries or want to raise an issue, please feel free to contact [Martin Weyssow](https://martin-wey.github.io/), [martin.weyssow@umontreal.ca](mailto:martin.weyssow@umontreal.ca).



## About

<div style="text-align: center;">

![Overview of CodeUltraFeedback](assets/CodeUltraFeedback.svg)

</div>

## 🧠 Models

| Model                             | Checkpoint                                                              | Size | CODAL-Bench - GPT-3.5<br/><span style="font-size:.8em;">(G-3.5, G-4)</span> | CODAL-Bench - GPT-4 <br/><span style="font-size:.8em;">G-4</span> | HumanEval (+)<br/><span style="font-size:.8em;">(k=1, k=10)</span> | License                                      |
|:----------------------------------|:------------------------------------------------------------------------|:----:|:---------------------------------------------------------------------------:|:-----------------------------------------------------------------:|:------------------------------------------------------------------:|:---------------------------------------------|
| **CodeLlama-7B-Instruct**         | 🤗 [HF Link](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) | `7B` |                                 6.00 / 5.46                                 |                               4.72                                |                            37.9 / 60.4                             | [Llama2](https://ai.meta.com/llama/license/) |
| **CodeLlama-7B-Instruct-SFT**     | 🤗 [HF Link](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) | `7B` |                                 6.51 / 5.83                                 |                               5.84                                |                            51.2 / 82.9                             | [Llama2](https://ai.meta.com/llama/license/) |
| **CodeLlama-7B-Instruct-DPO**     | 🤗 [HF Link](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) | `7B` |                                 7.15 / 6.79                                 |                               5.08                                |                            42.3 / 80.5                             | [Llama2](https://ai.meta.com/llama/license/) |
| **CodeLlama-7B-Instruct-SFT+DPO** | 🤗 [HF Link](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf) | `7B` |                                 7.36 / 7.08                                 |                               5.85                                |                            43.1 / 75.6                             | [Llama2](https://ai.meta.com/llama/license/) |

## 🤗 Datasets and Benchmark
- **CodeUltraFeedback**
- **CodeUltraFeedback-SFT**
- **CodeUltraFeedback-Binarized**
- **CODAL-Bench**