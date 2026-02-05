import torch
from typing import Iterable


def clip_gradient(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    Clip gradients of parameters to have a maximum L2 norm.

    Args:
        parameters: Iterable of parameters (may include params without gradients)
        max_norm: Maximum allowed L2 norm (M)
        eps: Small value for numerical stability

    Modifies parameter.grad in-place.

    Algorithm:
        1. Compute total L2 norm of all gradients: ||g||_2 = sqrt(sum of all grad²)
        2. If ||g||_2 > max_norm:
              scale each grad by max_norm / (||g||_2 + eps)
    """
    grads = []
    total_sq = 0.0
    for param in parameters:  # single iteration over parameters
        if param.grad is not None:
            grads.append(param.grad)
            total_sq += (param.grad ** 2).sum()

    l2 = total_sq ** 0.5
    if l2 > max_norm:
        scale_factor = max_norm / (l2 + eps)
        for grad in grads:
            grad.mul_(scale_factor)
