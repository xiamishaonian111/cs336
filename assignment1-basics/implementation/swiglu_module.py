import torch
import torch.nn as nn
from implementation.linear_module import Linear


class SwiGLU(nn.Module):

	def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
		"""Construct a SwiGLU feed-forward network.

		Args:
			d_model: int Dimensionality of the feedforward input and output
			d_ff: int Dimensionality of the inner feed-forward layer
			device: torch.device | None = None Device to store the parameters on
			dtype: torch.dtype | None = None Data type of the parameters
		"""
		super().__init__()
		self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
		self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
		self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype) 
		

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Apply SwiGLU feed-forward transformation.

		SwiGLU(x) = W2 @ (SiLU(W1 @ x) * (W3 @ x))

		Args:
			x: torch.Tensor input of shape (... , d_model)

		Returns:
			torch.Tensor output of shape (... , d_model)
		"""
		w1x = self.w1(x)
		silu = w1x * torch.sigmoid(w1x)
		return self.w2(silu * self.w3(x))
