# Source: https://github.com/evalplus/evalplus/blob/master/codegen/generate.py
# Slightly adapted for Transformers and AutoGPTQ libraries

import argparse
import os
import re

import torch
from auto_gptq import AutoGPTQForCausalLM, exllama_set_max_input_length
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from templates import templates


def construct_contract_prompt(prompt: str, contract_type: str, contract: str) -> str:
    if contract_type == "none":
        return prompt
    elif contract_type == "docstring":
        # embed within the docstring
        sep = ""
        if '"""' in prompt:
            sep = '"""'
        elif "'''" in prompt:
            sep = "'''"
        assert sep != ""
        l = prompt.split(sep)
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        l[1] = (
                l[1] + contract + "\n" + " " * (len(contract) - len(contract.lstrip()) - 1)
        )
        return sep.join(l)
    elif contract_type == "code":
        # at the beginning of the function
        contract = "\n".join([x.split("#")[0] for x in contract.splitlines()])
        return prompt + contract


def code_generate(args, workdir: os.PathLike, model, id_range=None):
    with Progress(
            TextColumn(
                f"{args.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"
            ),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
    ) as p:
        if args.dataset == "humaneval":
            from evalplus.data import get_human_eval_plus

            dataset = get_human_eval_plus()
        elif args.dataset == "mbpp":
            from evalplus.data import get_mbpp_plus

            dataset = get_mbpp_plus()

        for task_id, task in p.track(dataset.items()):
            if id_range is not None:
                id_num = int(task_id.split("/")[1])
                low, high = id_range
                if id_num < low or id_num >= high:
                    p.console.print(f"Skipping {task_id} as it is not in {id_range}")
                    continue

            p_name = task_id.replace("/", "_")
            if args.contract_type != "none" and task["contract"] == "":
                continue
            os.makedirs(os.path.join(workdir, p_name), exist_ok=True)
            log = f"Codegen: {p_name} @ {args.model}"
            n_existing = 0
            if args.resume:
                # count existing .py files
                n_existing = len(
                    [
                        f
                        for f in os.listdir(os.path.join(workdir, p_name))
                        if f.endswith(".py")
                    ]
                )
                if n_existing > 0:
                    log += f" (resuming from {n_existing})"

            nsamples = args.n_samples - n_existing
            p.console.print(log)

            sidx = args.n_samples - nsamples
            while sidx < args.n_samples:
                prompt = templates[args.model][args.dataset].format(prompt=construct_contract_prompt(
                    task["prompt"], args.contract_type, task["contract"]))
                responses = model(prompt,
                                  max_new_tokens=512,
                                  temperature=args.temperature,
                                  do_sample=True,
                                  top_p=0.95,
                                  num_return_sequences=args.n_samples,
                                  return_full_text=False)
                pattern = r"\[PYTHON\](.*?)\[/PYTHON\]"
                responses_trimmed = []
                for response in responses:
                    codes = re.findall(pattern, response['generated_text'], re.DOTALL)
                    responses_trimmed.append(codes[0].strip().replace('\t', '    ') if codes else '')

                for response in responses_trimmed:
                    try:
                        with open(
                                os.path.join(workdir, p_name, f"{sidx}.py"),
                                "w",
                                encoding="utf-8",
                        ) as f:
                            f.write(response)
                    except UnicodeEncodeError:
                        continue
                    sidx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument(
        "--dataset", required=True, type=str, choices=["humaneval", "mbpp"]
    )
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--n_samples", default=1, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--contract-type",
        default="none",
        type=str,
        choices=["none", "code", "docstring"],
    )
    parser.add_argument("--greedy", action="store_true")
    # id_range is list
    parser.add_argument("--id-range", default=None, nargs="+", type=int)
    args = parser.parse_args()

    if args.greedy and (args.temperature != 0 or args.bs != 1 or args.n_samples != 1):
        args.temperature = 0
        args.bs = 1
        args.n_samples = 1
        print("Greedy decoding ON (--greedy): setting bs=1, n_samples=1, temperature=0")

    if args.id_range is not None:
        assert len(args.id_range) == 2, "id_range must be a list of length 2"
        assert args.id_range[0] < args.id_range[1], "id_range must be increasing"
        args.id_range = tuple(args.id_range)

    # Make project dir
    os.makedirs(args.root, exist_ok=True)
    # Make dataset dir
    os.makedirs(os.path.join(args.root, args.dataset), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    # model = AutoGPTQForCausalLM.from_quantized(args.model_dir, device_map='auto', torch_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map='auto', torch_dtype=torch.float16)
    model = exllama_set_max_input_length(model, max_input_length=8192)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

    workdir = os.path.join(
        args.root,
        args.dataset,
        args.model
        + f"_temp_{args.temperature}"
        + ("" if args.contract_type == "none" else f"-contract-{args.contract_type}"),
    )
    os.makedirs(workdir, exist_ok=True)

    with open(os.path.join(workdir, "args.txt"), "w") as f:
        f.write(str(args))

    code_generate(args, workdir=workdir, model=generator, id_range=args.id_range)