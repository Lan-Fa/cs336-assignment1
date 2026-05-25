from __future__ import annotations

import numpy.typing as npt
import numpy.random as npr
import torch

def run_get_batch(dataset, batch_size: int, context_length: int, device: str):
    import numpy as np
    import torch

    max_start = len(dataset) - context_length - 1
    starts = np.random.randint(0, max_start + 1, size=(batch_size,))
    offsets = starts[:, None] + np.arange(context_length)[None, :]

    x_np = dataset[offsets].astype(np.int64, copy=False)
    y_np = dataset[offsets + 1].astype(np.int64, copy=False)

    x = torch.from_numpy(x_np).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.long)

    return x, y