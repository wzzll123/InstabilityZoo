import json
import subprocess
import argparse
import os
from pathlib import Path

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def build_command(args, config):
    json_model_to_config_file = {'GPT2-S':'gpt2_small', 'GPT2-M':'gpt2_medium', 'GPT2-L':'gpt2_large', 'GPT2-XL':'gpt2_xl'}
    config_file_model = json_model_to_config_file[config['model']]
    base_cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={args.nproc}", "reproduce_instability.py",
        f"config/reproduce_{config_file_model}.py",
        f"--ckpt_path={args.checkpoint}",
        f'--indices_path={args.index_file}',
        f"--learning_rate={config['learning_rate']}",
        f"--weight_decay={config['decay']}",
        f"--warmup_iters={config['warm']}",
        f"--out_dir={args.out_dir}",
        f"--start_idx={args.start_idx}",
        f"--max_iters={args.end_idx}",
    ]
    return base_cmd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--nproc", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.metadata)
    cmd = build_command(args, config)
    
    checkpoint_name = args.checkpoint.split('/')[-1]
    os.makedirs(args.out_dir, exist_ok=True)
    log_file = Path(args.out_dir) / f"reproduce-{checkpoint_name}-iter-{args.start_idx}.log"
    with open(log_file, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

if __name__ == "__main__":
    main()

    