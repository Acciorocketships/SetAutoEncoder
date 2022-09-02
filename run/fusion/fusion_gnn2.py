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

	def forward(self, x: Tensor, edge_index: Tensor, posx: Tensor, posa: Tensor):
		edge_index = edge_index.flip(dims=(0,)) # switch from (agents -> objects) to (objects -> agents)
		return self.propagate(x=x, posx=posx, posa=posa, edge_index=edge_index, size=(posx.shape[0], posa.shape[0]))

	def message(self, x_j: Tensor, posx_j: Tensor, posa_i: Tensor) -> Tensor:
		if self.position == 'abs':
			pos = posx_j
		elif self.position == 'rel':
			pos = posx_j - posa_i
		else:
			return x_j
		msg = torch.cat([x_j, pos], dim=-1)
		return msg

	def aggregate(self, inputs: Tensor, index: Tensor,
				  ptr: Optional[Tensor] = None,
				  dim_size: Optional[int] = None) -> Tensor:
		out = self.encoder(inputs, batch=index, n_batches=dim_size)
		return out


class MergeGNN(MessagePassing):
	def __init__(self, autoencoder, position='abs', **kwargs):
		super().__init__()
		self.position = position
		self.autoencoder = autoencoder
		self.merge_decoder = autoencoder.decoder
		self.merge_encoder = autoencoder.encoder
		self.filter = FilterModel(input_dim=self.merge_encoder.input_dim, hidden_dim=self.merge_encoder.input_dim)
		self.reset_values()

	def reset_values(self):
		self.values = []

	def get_values(self, key):
		return self.values[key]

	def forward(self, x: Tensor, edge_index: Tensor, pos: Tensor):
		edge_index = self.sort_edge_index(edge_index)
		self.set_decoder_preds(x, edge_index)
		return self.propagate(x=x, edge_index=edge_index, pos=pos, idx=torch.arange(pos.shape[0]).unsqueeze(1), size=(x.shape[0], x.shape[0]))

	def sort_edge_index(self, edge_index): # TODO: is this necessary?
		perm = torch.argsort(edge_index[1, :], stable=True)
		return edge_index[:,perm]

	def set_decoder_preds(self, x: Tensor, edge_index: Tensor):
		decoded, decoded_batch = self.merge_decoder(x)
		self.values.append(self.autoencoder.get_vars())

	def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:  # pos_j: edge_index[0,:], pos_i: edge_index[1,:]
		decoded, decoded_batch = self.merge_decoder(x_j) # = self.merge_decoder(x[edge_index[1,:],:])
		if self.position == 'rel':
			ope = scatter(src=torch.ones(decoded_batch.shape[0]), index=decoded_batch, dim_size=idx_i.shape[0]).long()
			pos_i_exp = pos_i.repeat_interleave(ope, dim=0)
			pos_j_exp = pos_j.repeat_interleave(ope, dim=0)
			decoded = self.update_rel_pos(decoded, pos_i_exp, pos_j_exp)
		return (decoded, decoded_batch)

	def update_rel_pos(self, x: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
		pos_dim = 2 if self.position is not None else 0
		x[:,-pos_dim:] = x[:,-pos_dim:] + pos_j - pos_i
		return x

	def filter_duplicates(self, x: Tensor, index: Tensor, dim_size: Optional[int] = None):
		mask = torch.zeros(x.shape[0], dtype=bool)
		if dim_size is None:
			if len(index) > 0:
				dim_size = max(index) + 1
			else:
				dim_size = 0
		for i in range(dim_size):
			xs = x[index == i, :]
			xi = xs.unsqueeze(0).expand(xs.shape[0], -1, -1)
			xj = xs.unsqueeze(1).expand(-1, xs.shape[0], -1)
			feat_mat = torch.cat([xi, xj], dim=2)
			feat_mat_flat = feat_mat.reshape(xs.shape[0]**2, 2*xs.shape[1])
			class_mat_flat = self.filter(feat_mat_flat)
			class_mat = class_mat_flat.reshape(xs.shape[0], xs.shape[0], 2)
			class_mat[:,:,1] *= 4
			class_mat_binary = torch.argmax(class_mat, dim=2)
			duplicates = torch.triu(class_mat_binary, diagonal=1)
			maski = duplicates.sum(dim=0) == 0
			if maski.sum() >= self.merge_encoder.max_n:
				idxs = torch.where(maski)[0][self.merge_encoder.max_n - 1:]
				maski[idxs] = False
			mask[index == i] = maski
		return mask

	def aggregate(self, inputs: Tensor, index: Tensor,
				  ptr: Optional[Tensor] = None,
				  dim_size: Optional[int] = None) -> Tensor:
		x = inputs[0]
		agent_idx = index[inputs[1]] # index: the agent of each edge, inputs[1]: the edge of each object.
		mask = self.filter_duplicates(x, agent_idx)
		x = x[mask,:]
		agent_idx = agent_idx[mask]
		output = self.merge_encoder(x, batch=agent_idx, n_batches=dim_size)
		return output


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

# TODO: make sure it works when position=None, and pos is not passed to forward
# TODO: make sure that "rel" position works