import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample random input sequences and next-token targets from a dataset.

    Args:
        dataset: 1D numpy array of integer token IDs.
        batch_size: Number of sequences to sample.
        context_length: Length of each sampled sequence.
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0').

    Returns:
        Tuple of (x, y), both LongTensors of shape (batch_size, context_length).
        x contains input token IDs, y contains the corresponding next-token targets
        (offset by 1 position from x).
    """
    start_indices = np.random.randint(0, len(dataset) - context_length, size=batch_size)

    # Load the sampled slices into tensors on the requested device.
    x = torch.stack(
        [torch.from_numpy(dataset[i : i + context_length].astype(np.int64)) for i in start_indices]
    ).to(device)
    y = torch.stack(
        [torch.from_numpy(dataset[i + 1 : i + 1 + context_length].astype(np.int64)) for i in start_indices]
    ).to(device)
    return x, y
