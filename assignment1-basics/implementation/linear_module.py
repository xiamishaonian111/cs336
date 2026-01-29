import torch
import torch.nn as nn
from einops import rearrange, einsum

class Linear(nn.models):

	def __init__(self, in_features, out_features, device=None, dtype=None):
		"""
		Construct a linear transformation module.

		Args:
			in_features: int final dimension of the input
			out_features: int final dimension of the output
			device: torch.device | None = None Device to store the parameters on
			dtype: torch.dtype | None = None Data type of the parameters
		"""
		super().__init__()
		self.weights = nn.Parameter(torch.empty(out_features, in_features))
		self.in_features = in_features
		self.out_features = out_features
		self.device = device
		self.dtype = dtype

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Apply the linear transformation to the input.

		Args:
		    x: torch.Tensor input matrix>
		"""
		x_in_features_last = rearrange(x, "... -> ... d_in")
		return einsum(x_in_features_last, weights, "... in_features, out_features in_features")
