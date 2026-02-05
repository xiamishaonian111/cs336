import math
import torch
from torch import Tensor
from typing import Iterable


class AdamW(torch.optim.Optimizer):

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]

                state["m"] = beta1 * state["m"] + (1 - beta1) * grad
                state["v"] = beta2 * state["v"] + (1 - beta2) * grad ** 2

                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * state["m"] / (torch.sqrt(state["v"]) + eps)
                p.data -= lr * weight_decay * p.data

        return loss
