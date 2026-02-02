import torch
import torch.nn as nn
from torch import Tensor

from implementation.multihead_self_attention import MultiheadSelfAttention
from implementation.rms_normalization import RMSNorm
from implementation.rope_module import RotaryPositionalEmbedding
from implementation.swiglu_module import SwiGLU


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        """
        Pre-norm Transformer block.

        Args:
            d_model: int Dimensionality of the input and output.
            num_heads: int Number of attention heads.
            d_ff: int Dimensionality of the feed-forward inner layer.
        """
        super().__init__()
        self.rms1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads)
        self.rms2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(
        self, x: Tensor, rope: RotaryPositionalEmbedding | None = None, token_positions: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            rope: Optional RoPE module to apply in attention
            token_positions: Optional tensor of shape (batch, seq_len) for RoPE

        Returns:
            Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.attn(self.rms1(x), rope, token_positions)
        x = x + self.ffn(self.rms2(x))
        return x
        
