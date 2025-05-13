total_batch_size = 524288 # 2**19 ~0.5M
batch_size = 64
block_size = 1024
gradient_accumulation_steps = total_batch_size // (batch_size*block_size)

# this makes total number of tokens be 300B
max_iters = 100000
# warmup_iters = 715
lr_decay_iters = 600000

# learning_rate = 1e-2
min_lr = 1e-5

log_max_logits = False
log_mean_logits = False
log_adam_m_v = False

n_layer = 12
n_head = 12
n_embd = 768

# out_dir = f'out_lr_{learning_rate}_decay_{weight_decay}_layer_{n_layer}_reproduce'

