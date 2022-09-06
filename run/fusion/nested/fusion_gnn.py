from torch import Tensor
import torch
from graphtorch import MessagePassing
from graphtorch.util import *
from sae.mlp import build_mlp
from sae.util import *




class MergeGNN(MessagePassing):
	def __init__(self, autoencoder, position='abs', **kwargs):
		super().__init__()
		self.position = position
		self.autoencoder = autoencoder
		self.decoder = autoencoder.decoder
		self.encoder = autoencoder.encoder
		self.filter = FilterModel(input_dim=self.encoder.input_dim, hidden_dim=self.encoder.input_dim)

	def forward(self, z: Tensor, edge_index: Tensor, **kwargs):
		# Decode
		xr_flat, batchr = self.decoder(z)
		nr = self.decoder.get_n_pred()
		xr = create_nested(xr_flat, nr)
		obj_idx = truncate_nested(kwargs["obj_idx"], nr)
		# Message
		print("shape z", z.shape)
		print("shape xr", shape_nested(xr))
		print("max edge_idx gnn", torch.max(edge_index))
		xr_neighbours = self.propagate(xr, edge_index)
		breakpoint()



class Encoder(torch.nn.Module):

	def __init__(self, autoencoder, position='abs', **kwargs):
		super().__init__()
		self.position = position
		self.encoder = autoencoder.encoder


	def forward(self, x: Tensor, **kwargs):

		x_flat, batch = nested_to_batch(x)
		num_agents = size_nested(x, dim=0).item()

		if self.position == "rel":
			assert ("pos" in kwargs), "In 'rel' mode, the input dict must contain 'pos', the agents' positions"
			pos = kwargs["pos"]
			pos_flat = pos[batch]
			x_flat[:,-pos_flat.shape[-1]:] = x_flat[:,-pos_flat.shape[-1]:] - pos_flat

		z = self.encoder(x_flat, batch, n_batches=num_agents)
		out = {}
		out["z"] = z

		if "obj_idx" in kwargs:
			perm = self.encoder.get_x_perm()
			obj_idx = permute_nested(kwargs["obj_idx"], perm)
			out["obj_idx"] = obj_idx

		return out




class Decoder(torch.nn.Module):

	def __init__(self, autoencoder, position='abs', **kwargs):
		super().__init__()
		self.position = position
		self.decoder = autoencoder.decoder




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