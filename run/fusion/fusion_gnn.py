from torch_scatter import scatter
from torch import Tensor
import torch
from torch_geometric.nn import MessagePassing
from typing import Optional
from sae import EncoderNew, DecoderNew
from sae import batch_to_set_lens


class EncodeGNN(MessagePassing):
	def __init__(self, in_channels, out_channels, max_obj=8, position='abs', **kwargs):
		super().__init__()
		self.input_dim = in_channels
		self.output_dim = out_channels
		self.position = position
		self.encoder = EncoderNew(dim=self.input_dim, hidden_dim=self.output_dim, max_n=max_obj)

	def forward(self, x: Tensor, edge_index: Tensor, posx: Tensor, posa: Tensor):
		edge_index = edge_index.flip(dims=(0,))
		self.agent_pos = posa
		self.obj_pos = posx
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
		self.filter_dist_thres = 1e-8
		self.input_decoder = DecoderNew(hidden_dim=self.input_dim, dim=self.orig_dim, max_n=max_obj)
		self.merge_encoder = EncoderNew(dim=self.orig_dim, hidden_dim=self.output_dim, max_n=max_obj)
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
			"x_idx_per_edge": [],
		}

	def get_values(self, key):
		return self.values[key]

	def forward(self, x: Tensor, edge_index: Tensor,  pos: Tensor):
		self.set_decoder_preds(x, edge_index)
		return self.propagate(x=x, edge_index=edge_index, pos=pos, size=(x.shape[0], x.shape[0]))

	def forward_true(self, x: Tensor, agent_idx: Tensor, obj_idx: Tensor, edge_index: Tensor, pos: Tensor):
		xe, xe_idx, obje_idx = self.agents_to_edges(x=x, agent_idx=agent_idx, edge_index=edge_index, obj_idx=obj_idx)
		pos_j = pos[edge_index[0, :], :]
		pos_i = pos[edge_index[1, :], :]
		if self.position == 'rel':
			xe = self.update_rel_pos(xe, xe_idx, pos_j, pos_i)
		mask = self.filter_duplicates(xe, xe_idx)
		x_new = xe[mask,:]
		agent_idx_new = xe_idx[mask]
		obj_idx_new = obje_idx[mask]
		return x_new, agent_idx_new, obj_idx_new

	def agents_to_edges(self, x, agent_idx, edge_index, obj_idx=None):
		# input: x, data for a variable number of objects for each agent
		#		 x_idx, the agent associated with each object
		#		 edge_index: the edge index of the graph (edges from edge_index[0,:] to edge_index[1,:])
		# purpose: duplicate x onto each incoming edge, and change x_idx to point at edges instead of agents
		perm = torch.argsort(agent_idx, stable=True)
		x = x[perm,:]
		agent_idx = agent_idx[perm]
		obj_idx = obj_idx[perm] if (obj_idx is not None) else None
		n_agent_idx = torch.max(agent_idx, dim=0)[0] + 1 if agent_idx.shape[0] > 0 else 0
		n_edge_index = torch.max(edge_index[0,:], dim=0)[0] + 1
		n_agents = max(n_edge_index, n_agent_idx)
		epa = scatter(src=torch.ones(edge_index.shape[1]), index=edge_index[0, :], dim_size=n_agents).long()
		opa = scatter(src=torch.ones(agent_idx.shape[0]), index=agent_idx, dim_size=n_agents).long()
		epa_cum = torch.cat([torch.zeros(1), torch.cumsum(epa, dim=0)], dim=0).long()
		opa_cum = torch.cat([torch.zeros(1), torch.cumsum(opa, dim=0)], dim=0).long()
		xn_idx_edge = torch.cat([torch.arange(epa[i]).repeat_interleave(opa[i]) + epa_cum[i] for i in range(epa.shape[0])],dim=0).long()
		agent_idx_new = edge_index[1,xn_idx_edge]
		src_idx = edge_index[0,xn_idx_edge]
		idx = torch.cat([torch.arange(opa[i]).repeat(epa[i]) + opa_cum[i] for i in range(epa.shape[0])],dim=0).long()
		x_new = x[idx]
		if obj_idx is not None:
			obj_idx_new = obj_idx[idx]
			return x_new, agent_idx_new, obj_idx_new
		else:
			return x_new, agent_idx_new

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
			x_pe, x_idx_pe, obj_idx_pe = self.agents_to_edges(x=decoded, agent_idx=decoded_batch, edge_index=edge_index, obj_idx=obj_idx_decoded)
			self.values["obj_idx_per_edge"].append(obj_idx_pe)
			self.values["x_per_edge"].append(x_pe)
			self.values["x_idx_per_edge"].append(x_idx_pe)
		self.values["n_pred_logits"].append(self.input_decoder.get_n_pred_logits())
		self.values["n_pred"].append(self.input_decoder.get_n_pred())
		self.values["x_pred"].append(decoded)
		self.values["batch_pred"].append(decoded_batch)

	def message(self, x_j: Tensor, pos_j: Tensor, pos_i: Tensor) -> Tensor:  # pos_j: edge_index[0,:], pos_i: edge_index[1,:]
		decoded, decoded_batch = self.input_decoder(x_j) # = self.input_decoder(x[edge_index[1,:],:])
		if self.position == 'rel':
			decoded = self.update_rel_pos(decoded, decoded_batch, pos_j, pos_i)
		return (decoded, decoded_batch)

	def update_rel_pos(self, decoded: Tensor, decoded_batch: Tensor, pos_j: Tensor, pos_i: Tensor) -> Tensor:
		decoded_pos = decoded[:,-self.pos_dim:]
		objs_per_agent = batch_to_set_lens(decoded_batch, pos_j.shape[0])
		relposij = torch.repeat_interleave((pos_j - pos_i), objs_per_agent, dim=0) # posj - posi
		new_pos = decoded_pos + relposij
		decoded[:,-self.pos_dim:] = new_pos
		return decoded

	def filter_duplicates(self, x: Tensor, index: Tensor, dim_size: Optional[int] = None, pos_only: bool = False):
		if pos_only:
			feat = x[:, -self.pos_dim:]
		else:
			feat = x
		mask = torch.zeros(x.shape[0], dtype=bool)
		if dim_size is None:
			dim_size = max(index)+1
		for i in range(dim_size):
			agent_obj_feat = feat[index == i,:]
			obj_pairwise_disp = (agent_obj_feat.unsqueeze(0) - agent_obj_feat.unsqueeze(1)).norm(dim=-1)
			duplicates = torch.triu(obj_pairwise_disp < self.filter_dist_thres, diagonal=1)
			maski = duplicates.sum(dim=0) == 0
			if maski.sum() >= self.merge_encoder.max_n:
				idxs = torch.where(maski)[0][self.merge_encoder.max_n-1:]
				maski[idxs] = False
			mask[index == i] = maski
		return mask


	def aggregate(self, inputs: Tensor, index: Tensor,
				  ptr: Optional[Tensor] = None,
				  dim_size: Optional[int] = None) -> Tensor:
		x = inputs[0]
		obj_index = index[inputs[1]] # index: the agent of each edge, inputs[1]: the edge of each object.
		mask = self.filter_duplicates(x, obj_index) #, dim_size=dim_size)
		x = x[mask,:]
		obj_index = obj_index[mask]
		output = self.merge_encoder(x, batch=obj_index, n_batches=dim_size)
		self.values["n_output"].append(scatter(src=torch.ones(obj_index.shape[0]), index=obj_index, dim_size=dim_size, reduce='sum').long())
		self.values["x_output"].append(x)
		self.values["batch_output"].append(obj_index)
		self.values["perm_output"].append(self.merge_encoder.get_x_perm())
		if len(self.values["obj_idx_per_edge"]) > 0:
			obj_idx_per_edge = self.values["obj_idx_per_edge"][-1]
			obj_idx = obj_idx_per_edge[mask]
			self.values["obj_idx"].append(obj_idx)
		return output
