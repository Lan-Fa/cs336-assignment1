from __future__ import annotations

from typing import Any

import math
import torch

class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        default = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay
        }
        super().__init__(params, default)

    def step(self, closure: None = None):
        with torch.no_grad():
            for group in self.param_groups:
                lr = group['lr']
                [beta1, beta2] = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']

                for param in group['params']:
                    if param.grad is None:
                        continue
                    if 'step' not in self.state[param]:
                        self.state[param]['step'] = 0
                        self.state[param]['m'] = torch.zeros_like(param)
                        self.state[param]['v'] = torch.zeros_like(param)

                    self.state[param]['step'] += 1
                    self.state[param]['m'] = beta1 * self.state[param]['m'] + (1 - beta1) * param.grad
                    self.state[param]['v'] = beta2 * self.state[param]['v'] + (1 - beta2) * torch.pow(param.grad, 2)

                    m = self.state[param]['m'] / (1 - beta1 ** self.state[param]['step'])
                    v = self.state[param]['v'] / (1 - beta2 ** self.state[param]['step'])

                    tmp = lr * weight_decay * param
                    param.sub_(tmp)
                    tmp = lr * (m / (torch.sqrt(v) + eps))
                    param.sub_(tmp)
        return

def get_adamw_cls() -> Any:
    """
    Returns a torch.optim.Optimizer that implements AdamW.
    """
    return AdamWOptimizer

def run_get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """

    if it < warmup_iters:
        res = it / warmup_iters * max_learning_rate
    elif warmup_iters <= it <= cosine_cycle_iters:
        PI = math.acos(-1)
        need = cosine_cycle_iters - warmup_iters
        now = it - warmup_iters
        res = (math.cos(now / need * PI) + 1) * 0.5 * (max_learning_rate - min_learning_rate) + min_learning_rate
    else :
        res = min_learning_rate

    return res