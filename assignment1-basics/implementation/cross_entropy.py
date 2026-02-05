import torch
from torch import Tensor


def cross_entropy(x: Tensor, targets: Tensor) -> Tensor:
    """Compute the average cross-entropy loss.

    Args:
        x: Float tensor of shape (batch_size, vocab_size) with unnormalized logits.
        targets: Int tensor of shape (batch_size,) with target class indices.

    Returns:
        Scalar tensor with the average cross-entropy loss.
    """
    x = x - x.max(dim=-1, keepdim=True).values
    # log(a/b) = log(a) - log(b)
    log_sum_exp = torch.log(torch.sum(torch.exp(x), dim=-1))
    log_softmax_exp = x[torch.arange(x.size(-2)), targets]
    return (log_sum_exp - log_softmax_exp).mean()
