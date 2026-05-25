from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.data import run_get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamWOptimizer, run_get_lr_cosine_schedule
from cs336_basics.nn_utils import run_gradient_clipping
from cs336_basics.checkpoint import run_save_checkpoint


DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

train_data = np.load(DATA_DIR / "tinystories_train_ids.npy", mmap_mode="r")
valid_data = np.load(DATA_DIR / "tinystories_valid_ids.npy", mmap_mode="r")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# vocab_size = 1000
# context_length = 128
# batch_size = 16
#
# d_model = 256
# num_layers = 4
# num_heads = 4
# d_ff = 1024
# rope_theta = 10000.0
#
# max_iters = 2000
# eval_interval = 100
# eval_iters = 20
#
# max_lr = 3e-4
# min_lr = 3e-5
# warmup_iters = 100
# cosine_cycle_iters = max_iters

vocab_size = 1000
context_length = 256
batch_size = 16

d_model = 256
num_layers = 4
num_heads = 4
d_ff = 1024
rope_theta = 10000

max_iters = 20000
eval_interval = 500
eval_iters = 50

max_lr = 3e-4
min_lr = 3e-5
warmup_iters = 1000
cosine_cycle_iters = max_iters

model = TransformerLM(
    vocab_size=vocab_size,
    context_length=context_length,
    d_model=d_model,
    num_layers=num_layers,
    num_heads=num_heads,
    d_ff=d_ff,
    rope_theta=rope_theta,
    device=device,
)

optimizer = AdamWOptimizer(
    model.parameters(),
    lr=max_lr,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)


@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {}

    for split, data in [("train", train_data), ("valid", valid_data)]:
        total = 0.0
        for _ in range(eval_iters):
            x, y = run_get_batch(data, batch_size, context_length, device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1),
            )
            total += loss.item()

        losses[split] = total / eval_iters

    model.train()
    return losses


model.train()
start_time = time.time()

for it in range(max_iters):
    lr = run_get_lr_cosine_schedule(
        it=it,
        max_learning_rate=max_lr,
        min_learning_rate=min_lr,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )

    for group in optimizer.param_groups:
        group["lr"] = lr

    x, y = run_get_batch(train_data, batch_size, context_length, device)

    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        y.reshape(-1),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    run_gradient_clipping(model.parameters(), max_l2_norm=1.0)
    optimizer.step()

    if it % eval_interval == 0:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        print(
            f"iter {it:5d} | "
            f"train loss {losses['train']:.4f} | "
            f"valid loss {losses['valid']:.4f} | "
            f"lr {lr:.2e} | "
            f"elapsed {elapsed:.1f}s"
        )

    if it > 0 and it % 500 == 0:
        path = CKPT_DIR / f"tinystories_iter_{it}.pt"
        run_save_checkpoint(model, optimizer, it, path)
        print("saved checkpoint:", path)

path = CKPT_DIR / "tinystories_final.pt"
run_save_checkpoint(model, optimizer, max_iters, path)
print("saved final checkpoint:", path)