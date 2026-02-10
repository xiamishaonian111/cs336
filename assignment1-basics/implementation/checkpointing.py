import os
from typing import BinaryIO, IO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    """
    Serialize model, optimizer, and iteration state to disk.

    Args:
        model: The model whose state to save.
        optimizer: The optimizer whose state to save.
        iteration: The current training iteration number.
        out: Path or file-like object to write the checkpoint to.
    """
    to_save = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(to_save, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    """
    Restore model and optimizer state from a checkpoint.

    Args:
        src: Path or file-like object to load the checkpoint from.
        model: The model to restore state into (via load_state_dict).
        optimizer: The optimizer to restore state into (via load_state_dict).
            If None, optimizer state is skipped.

    Returns:
        The iteration number that was saved in the checkpoint.
    """
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
