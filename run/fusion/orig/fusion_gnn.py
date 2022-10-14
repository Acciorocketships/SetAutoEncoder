from torch_scatter import scatter
from torch import Tensor
import torch
from torch_geometric.nn import MessagePassing
from typing import Optional
from sae.sae_psi import Encoder, Decoder
from sae.mlp import build_mlp


class EncodeGNN(MessagePassing):
	def __init__(self, in_channels, out_channels, max_obj=8, position='abs', **kwargs):
		super().__init__()
		self.input_dim = in_channels
		self.output_dim = out_channels
		self.position = position
		self.encoder = Encoder(dim=self.input_dim, hidden_dim=self.output_dim, max_n=max_obj)

	def forward(self, x: Tensor, edge_index: Tensor, posx: Tensor, posa: Tensor):
		edge_index = edge_index.flip(dims=(0,)) # switch from (agents -> objects) to (objects -> agents)
		self.agent_pos = posa
		self.obj_pos = posx
		self.edge_index = edge_index
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
		self.input = inputs
		self.batch = index
		self.x_perm = self.encoder.get_x_perm()
		return out


class MergeGNN(MessagePassing):
	def __init__(self, in_channels, out_channels, orig_dim, max_obj=16, position='abs', **kwargs):
		super().__init__()
		self.input_dim = in_channels
		self.output_dim = out_channels
		self.pos_dim = 2 if position is not None else 0
		self.orig_dim = orig_dim
		self.position = position
		self.input_decoder = Decoder(hidden_dim=self.input_dim, dim=self.orig_dim, max_n=max_obj)
		self.merge_encoder = Encoder(dim=self.orig_dim, hidden_dim=self.output_dim, max_n=max_obj)
		self.filter = FilterModel(input_dim=self.orig_dim, hidden_dim=self.orig_dim)
		self.reset_values()

	def reset_values(self):
		self.values = {
			"n_pred_logits": [],
			"n_pred": [],
			"x_pred": [],
			"batch_pred": [],
			"n_output": [],
			"x_output": [],
			"batch_output": [],
			"perm_output": [],
			"max_n": self.input_decoder.max_n,
			"obj_idx": [],
			"obj_idx_per_edge": [],
			"x_per_edge": [],
			"agent_idx_per_edge": [],
		}

	def get_values(self, key):
		return self.values[key]

	def forward(self, x: Tensor, edge_index: Tensor, pos: Tensor, obj_idx: Optional[Tensor] = None):
		edge_index = self.sort_edge_index(edge_index)
		self.set_decoder_preds(x, edge_index)
		self.obj_idx = obj_idx
		return self.propagate(x=x, edge_index=edge_index, pos=pos, idx=torch.arange(pos.shape[0]).unsqueeze(1), size=(x.shape[0], x.shape[0]))

	def sort_edge_index(self, edge_index): # TODO: is this necessary?
		perm = torch.argsort(edge_index[1, :], stable=True)
		return edge_index[:,perm]

	def agents_to_edges(self, x, agent_idx, edge_index, obj_idx=None):
		# input: x, data for a variable number of objects for each agent
		#		 x_idx, the agent associated with each object
		#		 edge_index: the edge index of the graph (edges from edge_index[0,:] to edge_index[1,:])
		# purpose: duplicate x onto each incoming edge, and change x_idx to point at edges instead of agents
		perm = torch.argsort(agent_idx, stable=True)
		x = x[perm,:]
		agent_idx = agent_idx[perm]
		n_agent_idx = torch.max(agent_idx, dim=0)[0] + 1 if agent_idx.shape[0] > 0 else 0
		n_edge_index = torch.max(edge_index[0,:], dim=0)[0] + 1
		n_agents = max(n_edge_index, n_agent_idx)
		epa = scatter(src=torch.ones(edge_index.shape[1]), index=edge_index[0, :], dim_size=n_agents).long()
		opa = scatter(src=torch.ones(agent_idx.shape[0]), index=agent_idx, dim_size=n_agents).long()
		epa_cum = torch.cat([torch.zeros(1), torch.cumsum(epa, dim=0)], dim=0).long()
		opa_cum = torch.cat([torch.zeros(1), torch.cumsum(opa, dim=0)], dim=0).long()
		xn_idx_edge = torch.cat([torch.arange(epa[i]).repeat_interleave(opa[i]) + epa_cum[i] for i in range(epa.shape[0])], dim=0).long()
		agent_idx_new = edge_index[0, xn_idx_edge]
		agent_src_idx_new = edge_index[1, xn_idx_edge]
		idx = torch.cat([torch.arange(opa[i]).repeat(epa[i]) + opa_cum[i] for i in range(epa.shape[0])], dim=0).long()
		x_new = x[idx]
		obj_idx_new = None
		if obj_idx is not None:
			obj_idx = obj_idx[perm]
			obj_idx_new = obj_idx[idx]
		return {
			"x": x_new,
			"agent_idx": agent_idx_new,
			"agent_src_idx": agent_src_idx_new,
			"obj_idx": obj_idx_new,
		}

	def set_decoder_preds(self, x: Tensor, edge_index: Tensor):
		decoded, decoded_batch = self.input_decoder(x)
		if len(self.values["obj_idx"]) > 0:
			obj_idx = self.values["obj_idx"][-1]
			n_output = self.values["n_output"][-1]
			batch_ptr = torch.cat([torch.zeros(1), torch.cumsum(n_output, dim=0)], dim=0).long()
			n_pred = self.input_decoder.get_n_pred()
			obj_idx_decoded = torch.cat([
					torch.cat([
						obj_idx[batch_ptr[i]:min(batch_ptr[i]+n_pred[i], batch_ptr[i+1])],
						torch.full((max(n_pred[i] - n_output[i], 0),), torch.nan),
					], dim=0)
				for i in range(n_pred.shape[0])])
			edge_data = self.agents_to_edges(x=decoded, agent_idx=decoded_batch, edge_index=edge_index, obj_idx=obj_idx_decoded)
			self.values["obj_idx_per_edge"].append(edge_data["obj_idx"])
			self.values["x_per_edge"].append(edge_data["x"])
			self.values["agent_idx_per_edge"].append(edge_data["agent_idx"])
		self.values["n_pred_logits"].append(self.input_decoder.get_n_pred_logits())
		self.values["n_pred"].append(self.input_decoder.get_n_pred())
		self.values["x_pred"].append(decoded)
		self.values["batch_pred"].append(decoded_batch)

	def set_obj_idxs(self, obj_idxs, idx_i, idx_j, decoded, decoded_batch):
		obj_idx_edge = torch.nested_tensor([])
		breakpoint()


	def message(self, x_j: Tensor, pos_i: Tensor, pos_j: Tensor, idx_i: Tensor, idx_j: Tensor) -> Tensor:  # pos_j: edge_index[0,:], pos_i: edge_index[1,:]
		decoded, decoded_batch = self.input_decoder(x_j) # = self.input_decoder(x[edge_index[1,:],:])
		# self.set_obj_idxs(obj_idxs=self.obj_idx, idx_i=idx_i, idx_j=idx_j, decoded=decoded, decoded_batch=decoded_batch)
		if self.position == 'rel':
			ope = scatter(src=torch.ones(decoded_batch.shape[0]), index=decoded_batch, dim_size=idx_i.shape[0]).long()
			pos_i_exp = pos_i.repeat_interleave(ope, dim=0)
			pos_j_exp = pos_j.repeat_interleave(ope, dim=0)
			decoded = self.update_rel_pos(decoded, pos_i_exp, pos_j_exp)
		return (decoded, decoded_batch)

	def update_rel_pos(self, x: Tensor, pos_i: Tensor, pos_j: Tensor) -> Tensor:
		x[:,-self.pos_dim:] = x[:,-self.pos_dim:] + pos_j - pos_i
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
		self.values["n_output"].append(scatter(src=torch.ones(agent_idx.shape[0]), index=agent_idx, dim_size=dim_size, reduce='sum').long())
		self.values["x_output"].append(x)
		self.values["batch_output"].append(agent_idx)
		self.values["perm_output"].append(self.merge_encoder.get_x_perm())
		if len(self.values["obj_idx_per_edge"]) > 0:
			obj_idx_per_edge = self.values["obj_idx_per_edge"][-1]
			obj_idx = obj_idx_per_edge[mask]
			self.values["obj_idx"].append(obj_idx)
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
