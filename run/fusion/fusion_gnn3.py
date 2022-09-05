from torch_scatter import scatter
from torch import Tensor
import torch
from torch_geometric.nn import MessagePassing
from typing import Optional
from sae import EncoderNew, DecoderNew
from sae.mlp import build_mlp


class EncodeGNN(MessagePassing):
	def __init__(self, autoencoder, position='abs', **kwargs):
		super().__init__()
		self.position = position
		self.encoder = autoencoder.encoder

	def forward(self, x: Tensor, edge_index: Tensor):
		pass



class MergeGNN(MessagePassing):
	def __init__(self, autoencoder, position='abs', **kwargs):
		super().__init__()
		self.position = position
		self.autoencoder = autoencoder
		self.merge_decoder = autoencoder.decoder
		self.merge_encoder = autoencoder.encoder
		self.filter = FilterModel(input_dim=self.merge_encoder.input_dim, hidden_dim=self.merge_encoder.input_dim)

	def forward(self, x: Tensor, edge_index: Tensor, pos: Tensor):
		pass


class FilterModel(torch.nn.Module):

	def __init__(self, input_dim, hidden_dim):
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = hidden_dim
		self.pre_mlp = build_mlp(input_dim=input_dim, output_dim=hidden_dim, nlayers=2)
		self.post_mlp = build_mlp(input_dim=hidden_dim, output_dim=2, nlayers=2)

	def forward(self, x):
		x1 = x[:,:self.input_dim]
		x2 = x[:,self.input_dim:]
		a1 = self.pre_mlp(x1)
		a2 = self.pre_mlp(x2)
		b = a1 * a2
		c = self.post_mlp(b)
		return c