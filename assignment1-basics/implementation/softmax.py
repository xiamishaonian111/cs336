import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Apply softmax to the given dimension of a tensor.

    Args:
        x: Input tensor of arbitrary shape.
        dim: Dimension to apply softmax along.

    Returns:
        Tensor of the same shape with softmax applied along `dim`.
    """
    x_max = x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
