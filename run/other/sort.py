import torch
from torch import nn
from mlp import build_mlp
import torchsort

class Rank(nn.Module):

	def __init__(self, dim):
		super().__init__()
		self.dim = dim
		self.criterion = build_mlp(input_dim=self.dim, output_dim=1, nlayers=2, midmult=1., layernorm=False)
		self.indices = None

	def forward(self, input):
		a = self.criterion(input)
		x = input / input.detach() * a
		y = torchsort.soft_sort(x)
		self.indices = torch.sort(a.squeeze())[1]
		z = y * input.detach() / a.detach()
		return z


class AscSort(nn.Module):

	def __init__(self):
		super().__init__()
		self.indices = None

	def forward(self, input):
		a = -torch.sum(input, dim=-1)
		idx = torch.sort(a)[1]
		self.indices = idx
		return input[idx,:]


class DescSort(nn.Module):

	def __init__(self):
		super().__init__()
		self.indices = None

	def forward(self, input):
		a = -torch.sum(input, dim=-1)
		idx = torch.sort(a)[1]
		self.indices = idx
		return input[idx,:]


