import torch
import torch.nn as nn
from torch import Tensor

from implementation.embedding_module import Embedding
from implementation.linear_module import Linear
from implementation.rms_normalization import RMSNorm
from implementation.rope_module import RotaryPositionalEmbedding
from implementation.transformer_block import TransformerBlock


class TransformerLM(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        """
        Transformer language model.

        Args:
            vocab_size: int Size of the vocabulary.
            context_length: int Maximum sequence length.
            d_model: int Dimensionality of embeddings and sublayer outputs.
            num_layers: int Number of transformer blocks.
            num_heads: int Number of attention heads.
            d_ff: int Dimensionality of the FFN inner layer.
            rope_theta: float RoPE theta parameter.
        """
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Int tensor of shape (batch_size, seq_len) with token indices.

        Returns:
            Float tensor of shape (batch_size, seq_len, vocab_size) with logits.
        """
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, self.rope)
        return self.lm_head(self.ln_final(x))
