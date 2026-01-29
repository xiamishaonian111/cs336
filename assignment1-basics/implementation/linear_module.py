import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):

	def __init__(self, in_features, out_features, device=None, dtype=None):
		"""Construct a linear transformation module.

		Args:
			in_features: int final dimension of the input
			out_features: int final dimension of the output
			device: torch.device | None = None Device to store the parameters on
			dtype: torch.dtype | None = None Data type of the parameters
		"""
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.weights = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
		std = math.sqrt(2 / (in_features + out_features))
		nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3 * std, b=3 * std)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Apply the linear transformation to the input."""
		if x.shape[-1] != self.in_features:
			raise ValueError(
				f"Expected input last dimension to be {self.in_features}, got {x.shape[-1]}"
			)
		return einsum(x, self.weights, "... in_features, out_features in_features -> ... out_features")
