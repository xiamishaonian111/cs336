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
		super().

	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		"""Lookup the embedding vectors for the given token IDs."""
		raise NotImplementedError
