import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch import Tensor

from implementation.rope_module import RotaryPositionalEmbedding
from implementation.scaled_dot_product_attention import scaled_dot_product_attention


class MultiheadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int):
        """
        Causal multi-head self-attention.

        Args:
            d_model: int Dimensionality of the input and output.
            num_heads: int Number of attention heads. d_model must be divisible by num_heads.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, x: Tensor, rope: RotaryPositionalEmbedding | None = None, token_positions: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape (..., seq_len, d_model)
            rope: Optional RoPE module to apply to Q and K
            token_positions: Optional tensor of shape (..., seq_len) for RoPE

        Returns:
            Tensor of shape (..., seq_len, d_model)
        """
        h = self.num_heads
        seq_len = x.size(-2)

        Q = rearrange(self.q_proj(x), "... s (h d) -> ... h s d", h=h)
        K = rearrange(self.k_proj(x), "... s (h d) -> ... h s d", h=h)
        V = rearrange(self.v_proj(x), "... s (h d) -> ... h s d", h=h)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)

        causal_mask = ~torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)

        attn_out = scaled_dot_product_attention(Q, K, V, causal_mask)

        attn_out = rearrange(attn_out, "... h s d -> ... s (h d)")

        # Output projection
        return self.o_proj(attn_out)
