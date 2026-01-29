import torch
import torch.nn as nn


class Embedding(nn.Module):

	def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
		"""Construct an embedding module.

		Args:
			num_embeddings: int Size of the vocabulary
			embedding_dim: int Dimension of the embedding vectors, i.e., d_model
			device: torch.device | None = None Device to store the parameters on
			dtype: torch.dtype | None = None Data type of the parameters
		"""
		super().__init__()
		self.num_embeddings = num_embeddings
		self.embedding_dim = embedding_dim
		self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
		nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)

	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		"""Lookup the embedding vectors for the given token IDs."""
		return self.weights[token_ids]
