import argparse
import glob
import json
import os

from rich.console import Console
from rich.table import Table


preferences = {
    'Instruction-Following': 'cyan',
    'Code Readability': 'blue',
    'Code Complexity and Efficiency': 'green',
    'Coding Style': 'magenta',
    'Code Explanation': 'purple'
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_responses_dir", type=str, default="./runs/eval_results/greedy/single")
    parser.add_argument("--model_judge", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()

    results_files = glob.glob(args.model_responses_dir + '/*.jsonl')
    results_files = sorted(results_files, key=lambda x: os.path.basename(x))
    models_results = []
    for file_path in results_files:
        if not args.model_judge in os.path.basename(file_path):
            continue
        if args.model_name and not args.model_name in file_path:
            continue
        file_name = os.path.basename(file_path)
        model_name = file_name.split('_')[0]
        with open(file_path, 'r') as f:
            data = [json.loads(l)['rating'] for l in f]
            data = list(map(lambda ex: 0 if ex == 'N/A' else ex, data))
            data = [sum(data[i:i + 100]) / 100 for i in range(0, len(data), 100)]
            if len(data) == 0:
                continue
            data.append(sum(data) / len(data))
            data = ['{:.2f}'.format(i) for i in data]
            data.insert(0, model_name)
            models_results.append(data)

    table = Table(title="Coding Preferences Scores")
    table.add_column("Model")
    for k, v in preferences.items():
        table.add_column(k, style=v)
    table.add_column("Average")

    for results in models_results:
        table.add_row(results[0], results[1], results[2], results[3], results[4], results[5], results[6])

    console = Console()
    console.print(table)


if __name__ == '__main__':
    main()
