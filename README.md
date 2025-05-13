
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
Run the following command to resume training from a checkpoint. The output log and optimizer state will be saved in the out_dir directory.
```
python reproduce_main.py \
  --checkpoint checkpoints/ckpt_1.pt \
  --index_file indices/ckpt_1_indices.txt \
  --metadata metadata/ckpt_1.json \
  --start_idx 38000 \
  --end_idx 40000 \
  --out_dir output/run1
```
