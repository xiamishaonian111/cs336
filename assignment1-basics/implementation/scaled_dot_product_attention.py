import math

import torch
from einops import einsum
from torch import Tensor

from implementation.softmax import softmax


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """
    Compute scaled dot-product attention.

    Args:
        Q: Query tensor of shape (..., n, d_k)
        K: Key tensor of shape (..., m, d_k)
        V: Value tensor of shape (..., m, d_v)
        mask: Optional boolean mask of shape (n, m).
              True = attend, False = ignore.

    Returns:
        Tensor of shape (..., n, d_v)
    """
    d_k = Q.shape[-1]
    qk = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
    qk_scaled = qk / math.sqrt(d_k)
    if mask is not None:
        qk_scaled = qk_scaled.masked_fill(~mask, float("-inf"))
    return einsum(softmax(qk_scaled, dim=-1), V, "... n m, ... m d_v -> ... n d_v")
