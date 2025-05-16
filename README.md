
# InstabilityZoo

## install

```
pip install torch==2.5.1 numpy transformers datasets tiktoken wandb tqdm
pip install --no-build-isolation transformer_engine[pytorch]==1.13.0
```

## Prepare Data
```
python data/openwebtext/prepare.py
```

## Models and Checkpoints
All models and checkpoints have hosted in https://huggingface.co/zzwen/InstabilityZoo.

## Using InstabilityZoo
First, download the checkpoints and related files from Hugging Face. For example:
```
wget https://huggingface.co/zzwen/InstabilityZoo/resolve/main/checkpoints/ckpt_1.pt
wget https://huggingface.co/zzwen/InstabilityZoo/resolve/main/indices/ckpt_1_indices.txt
wget https://huggingface.co/zzwen/InstabilityZoo/resolve/main/metadata/ckpt_1.json
```

Then, run the following command to resume training from a checkpoint. The output log and optimizer state will be saved in the specified out_dir. Be sure to specify the number of GPUs with --nproc; currently, our program only supports single-machine setups.
```
python reproduce_main.py \
  --checkpoint checkpoints/ckpt_1.pt \
  --index_file indices/ckpt_1_indices.txt \
  --metadata metadata/ckpt_1.json \
  --start_idx 38000 \
  --end_idx 40000 \
  --out_dir output/run1 \
  --nproc 8
```
