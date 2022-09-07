import torch
from torch import Tensor
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
		self.max_n = self.encoder.get_max_n()

	def forward(self, z: Tensor, edge_index: Tensor, **kwargs):
		# Decode
		xr_flat, batchr = self.decoder(z)
		self.store_autoencoder_training_data()
		nr = self.decoder.get_n_pred()
		xr = create_nested(xr_flat, nr)
		obj_idx = truncate_nested(kwargs["obj_idx"], nr)
		# Message
		xr_neighbours = self.propagate(xr, edge_index)
		obj_idx_neighbours = self.propagate(obj_idx, edge_index)
		xr_neighbours = self.update_rel_pos(xr_neighbours, kwargs.get("pos", None))
		# Filter
		xr_masked, obj_idx_masked = self.apply_filter(xr_neighbours, obj_idx_neighbours)
		self.store_filter_training_data(xr_neighbours, obj_idx_neighbours)
		# Encode
		x_flat, batch = nested_to_batch(xr_masked)
		num_agents = size_nested(xr_masked, dim=0).item()
		z = self.encoder(x_flat, batch, n_batches=num_agents)
		# Return
		return {
			"z": z,
			"obj_idx": obj_idx,
		}

	def apply_filter(self, x, obj_idx):
		# x: N x K(var) x D
		perm = apply_nested(self.duplicate_mask, x)
		x_masked = permute_nested(x, perm)
		obj_idx_masked = permute_nested(obj_idx, perm)
		return x_masked, obj_idx_masked

	def duplicate_mask(self, xi):
		# xi: K x D
		K, D = xi.shape
		xi_0 = xi.unsqueeze(0).expand(K, -1, -1)
		xi_1 = xi.unsqueeze(1).expand(-1, K, -1)
		xi_mat = torch.cat([xi_0, xi_1], dim=0)
		xi_mat_flat = xi_mat.reshape(K ** 2, 2 * D)
		classes_flat = self.filter(xi_mat_flat)
		classes = classes_flat.view(K, K, 2)
		classes_bin = classes.argmax(dim=-1)
		diag = torch.triu(classes_bin, diagonal=1)
		mask = diag.sum(dim=0) == 0
		if mask.sum() > self.max_n:
			idxs = torch.where(mask)[0][self.max_n:]
			mask[idxs] = False
		return mask

	def update_rel_pos(self, x, pos):
		if self.position == "rel":
			assert pos is not None
			pos_padded = torch.zeros(pos.shape[0], 1, size_nested(x, dim=-1))
			pos_padded[:,0,-2:] = pos
			x_rel = add_nested(x, -pos_padded)
			return x_rel
		return x

	def store_autoencoder_training_data(self):
		self.ae_training_data = self.autoencoder.get_vars()

	def get_autoencoder_training_data(self):
		return self.ae_training_data

	def store_filter_training_data(self, x, obj_idx):
		classes = []
		labels = []
		for (xi, obj_idxi) in zip(x.unbind(), obj_idx.unbind()):
			K, D = xi.shape
			# labels
			obj_idxi_0 = obj_idxi.unsqueeze(0).expand(K, -1)
			obj_idxi_1 = obj_idxi.unsqueeze(1).expand(-1, K)
			obj_idxi_mat = torch.stack([obj_idxi_0, obj_idxi_1], dim=-1)
			obj_idxi_mat_flat = obj_idxi_mat.view(K**2, 2)
			labels_flat = obj_idxi_mat_flat[:,0] == obj_idxi_mat_flat[:,1]
			mask = ~torch.any(torch.isnan(obj_idxi_mat_flat), dim=-1)
			labels_flat = labels_flat[mask]
			labels.append(labels_flat)
			# classes
			xi_0 = xi.unsqueeze(0).expand(K, -1, -1)
			xi_1 = xi.unsqueeze(1).expand(-1, K, -1)
			xi_mat = torch.cat([xi_0, xi_1], dim=-1)
			xi_mat_flat = xi_mat.view(K ** 2, 2 * D)
			xi_mat_flat = xi_mat_flat[mask]
			classes_flat = self.filter(xi_mat_flat)
			classes.append(classes_flat)
		classes = torch.cat(classes, dim=0)
		labels = torch.cat(labels, dim=0).long()
		self.filter_training_data = {
			"classes": classes,
			"labels": labels,
		}

	def get_filter_training_data(self):
		return self.filter_training_data




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
		self.autoencoder = autoencoder
		self.decoder = autoencoder.decoder

	def forward(self, z: Tensor, **kwargs):
		xr_flat, batchr = self.decoder(z)
		self.store_autoencoder_training_data()
		nr = self.decoder.get_n_pred()
		xr = create_nested(xr_flat, nr)
		obj_idx = truncate_nested(kwargs["obj_idx"], nr)
		return {
			"x": xr,
			"obj_idx": obj_idx,
		}

	def store_autoencoder_training_data(self):
		self.ae_training_data = self.autoencoder.get_vars()

	def get_autoencoder_training_data(self):
		return self.ae_training_data




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