import torch
import torch.nn as nn
from einops import rearrange


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and precompute cos/sin buffers.

        Args:
            theta: float Θ value for the RoPE
            d_k: int Dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even"
        self.d_k = d_k

        # Compute 1 / Theta^{(2k-2)/d} for each pair k in {1, ..., d_k/2}
        # Exponents: 0, 2/d_k, 4/d_k, ..., (d_k-2)/d_k
        exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k  # shape (d_k/2,)
        theta_no_i = 1.0 / (theta ** exponents)  # shape (d_k/2,)

        # Positions from 0 to max_seq_len-1
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # shape (max_seq_len,)

        # Outer product: angles[i, k] = position_i * freq_k
        angles = torch.outer(positions, theta_no_i)  # shape (max_seq_len, num_pairs)

        self.register_buffer("cos_cache", angles.cos(), persistent=False)  # (max_seq_len, num_pairs)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)  # (max_seq_len, num_pairs)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to input tensor.

        Args:
            x: Tensor of shape (..., seq_len, d_k)
            token_positions: Tensor of shape (..., seq_len) with position indices

        Returns:
            Tensor of same shape as x with rotary embeddings applied.
        """
        # Gather cos/sin for the requested positions: (..., seq_len) -> (..., seq_len, num_pairs)
        cos = self.cos_cache[token_positions]  # (..., seq_len, num_pairs)
        sin = self.sin_cache[token_positions]  # (..., seq_len, num_pairs)

        # Reshape to pairs: (..., seq_len, d_k) -> (..., seq_len, d_k/2, 2)
        x_paired = rearrange(x, "... (p k) -> ... p k", k=2)

        # Build rotation matrix: (..., seq_len, d_k/2, 2, 2)
        R = torch.stack([
            torch.stack([cos, -sin], dim=-1),
            torch.stack([sin,  cos], dim=-1),
        ], dim=-2)

        # Apply rotation and flatten back: (..., seq_len, d_k/2, 2) -> (..., seq_len, d_k)
        rotated = torch.einsum("... i j, ... j -> ... i", R, x_paired)
        return rearrange(rotated, "... p k -> ... (p k)")
