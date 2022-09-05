import torch
from torch import nn
from torch_geometric.nn import Sequential
from sae import AutoEncoderNew
from sae import cross_entropy_loss, correlation
from fusion_gnn2 import EncodeGNN, MergeGNN


class FusionModel(nn.Module):
	def __init__(self, input_dim, embedding_dim=64, autoencoder=AutoEncoderNew, gnn_nlayers=1, max_obj=16, position='rel', **kwargs):
		super().__init__()
		self.gnn_nlayers = gnn_nlayers
		self.max_obj = max_obj
		self.position = position
		pos_dim = 2 if (self.position is not None) else 0
		self.autoencoder = autoencoder(dim=input_dim+pos_dim, hidden_dim=embedding_dim, max_n=max_obj)
		self.encode_gnn = EncodeGNN(autoencoder=self.autoencoder, position=self.position, **kwargs)
		self.merge_gnn = self.create_merge_gnn(**kwargs)
		self.decoder = self.autoencoder.decoder


	def create_merge_gnn(self, **kwargs):
		layers = []
		signatures = []
		merge_gnn_layer = MergeGNN(
			autoencoder=self.autoencoder,
			position=self.position,
			**kwargs,
		)
		for i in range(self.gnn_nlayers):
			layers.append(merge_gnn_layer)
			signatures.append("x, edge_index, *args, **kwargs -> x")
		gnn = Sequential("x, edge_index, *args, **kwargs", zip(layers, signatures))
		gnn.reset_values = gnn[0].reset_values
		gnn.get_values = gnn[0].get_values
		gnn.append_values = gnn[0].append_values
		return gnn


	def forward(self, data):
		# self.forward_true(data)

		obj_x = data['object'].x
		obj_pos = data['object'].pos
		agent_pos = data['agent'].pos
		obj_agent_edge_index = data[('agent', 'observe', 'object')].edge_index # [agent_idx, obj_idx]
		agent_edge_index = data[('agent', 'communicate', 'agent')].edge_index

		self.merge_gnn.reset_values()

		self.enc = self.encode_gnn(x=obj_x, edge_index=obj_agent_edge_index, posx=obj_pos, posa=agent_pos)

		obj_idx = obj_agent_edge_index[1, self.encode_gnn.encoder.get_x_perm()]
		self.merge_gnn[0].filter_values["obj_idx"].append(obj_idx)

		self.merged = self.merge_gnn(x=self.enc, edge_index=agent_edge_index, pos=agent_pos)

		self.decoded, self.batch = self.decoder(self.merged)

		self.merge_gnn.append_values(self.autoencoder.get_vars())

		return self.decoded, self.batch


	def forward_true(self, data):
		obj_x = data['object'].x
		obj_pos = data['object'].pos
		agent_pos = data['agent'].pos
		obj_agent_edge_index = data[('agent', 'observe', 'object')].edge_index  # [agent_idx, obj_idx]
		agent_edge_index = data[('agent', 'communicate', 'agent')].edge_index

		agent_idx = obj_agent_edge_index[0,:]
		obj_idx = obj_agent_edge_index[1,:]
		x = self.encode_gnn.message(obj_x[obj_idx,:], obj_pos[obj_idx,:], agent_pos[agent_idx,:])
		for i in range(self.gnn_nlayers):
			x, agent_idx, obj_idx = self.merge_gnn[i].forward_true(x=x, agent_idx=agent_idx, obj_idx=obj_idx, edge_index=agent_edge_index, pos=agent_pos)
		return x, agent_idx, obj_idx


	def loss(self):
		autoencoder_loss = self.autoencoder_loss()
		filter_loss = self.filter_loss(return_accuracy=True)
		loss = autoencoder_loss["ae_loss"] + 0.3 * filter_loss["filter_loss"]
		return {
			"loss": loss,
			**autoencoder_loss,
			**filter_loss,
		}


	def autoencoder_loss(self):
		vars_perlayer = self.merge_gnn.get_values()
		keys = None
		loss = None
		for vars in vars_perlayer:
			layer_loss = self.autoencoder.loss(vars)
			if keys is None:
				keys = layer_loss.keys()
				loss = {key: 0 for key in keys}
			for key in keys:
				loss[key] += layer_loss[key]
		for key in keys:
			loss[key] /= len(vars_perlayer)
		loss["ae_loss"] = loss["loss"]
		del loss["loss"]
		return loss



	def filter_loss(self, return_accuracy=False, num_layers=None):
		def cross(z):
			zi = z.unsqueeze(0).expand(z.shape[0], -1, -1)
			zj = z.unsqueeze(1).expand(-1, z.shape[0], -1)
			z_mat = torch.cat([zi, zj], dim=-1)
			z_mat_flat = z_mat.reshape(z.shape[0] ** 2, 2 * z.shape[1])
			return z_mat_flat
		losses = []
		accuracies = []
		corr_same = []
		corr_diff = []
		ratio_pred = []
		ratio_true = []
		if num_layers is None:
			num_layers = self.gnn_nlayers
		for i in range(num_layers):
			obj_idx = self.merge_gnn.get_values("obj_idx_per_edge")[i]
			agent_idx = self.merge_gnn.get_values("agent_idx_per_edge")[i]
			x = self.merge_gnn.get_values("x_per_edge")[i]
			num_agents = torch.max(agent_idx)
			dup_class = []
			xs_cross = []
			for agent in range(num_agents):
				idxs = (agent_idx == agent)
				xs = x[idxs, :]
				objs = obj_idx[idxs]
				xs_cross_i = cross(xs)
				objs_cross = cross(objs.unsqueeze(-1))
				nan_mask = ~torch.any(torch.isnan(objs_cross), dim=1)
				objs_cross = objs_cross[nan_mask]
				dup_class_i = (objs_cross[:,0] == objs_cross[:,1]).long()
				xs_cross_i = xs_cross_i[nan_mask]
				dup_class.append(dup_class_i)
				xs_cross.append(xs_cross_i)
			dup_class = torch.cat(dup_class, dim=0)
			xs_cross = torch.cat(xs_cross, dim=0)
			pred_class = self.merge_gnn[0].filter(xs_cross)
			layer_loss = torch.mean(cross_entropy_loss(pred_class, dup_class))
			accuracy = torch.sum(torch.argmax(pred_class, dim=1) == dup_class) / dup_class.shape[0]
			xs_same = xs_cross[dup_class.bool(),:]
			xs_diff = xs_cross[~dup_class.bool(),:]
			d = xs_same.shape[-1] // 2
			corr_same_layer = correlation(xs_same[:,:d], xs_same[:,d:])
			corr_diff_layer = correlation(xs_diff[:, :d], xs_diff[:, d:])
			losses.append(layer_loss)
			accuracies.append(accuracy)
			corr_same.append(corr_same_layer)
			corr_diff.append(corr_diff_layer)
			ratio_pred.append(torch.sum(torch.argmax(pred_class, dim=1)) / dup_class.shape[0])
			ratio_true.append(torch.sum(dup_class) / dup_class.shape[0])
		loss = sum(losses)
		acc = sum(accuracies) / len(accuracies)
		corr_same = sum(corr_same) / len(corr_same)
		corr_diff = sum(corr_diff) / len(corr_diff)
		ratio_pred = sum(ratio_pred) / len(ratio_pred)
		ratio_true = sum(ratio_true) / len(ratio_true)
		if return_accuracy:
			return {
				"filter_loss": loss,
				"filter_acc": acc,
				"corr_same": corr_same,
				"corr_diff": corr_diff,
				"ratio_pred": ratio_pred,
				"ratio_true": ratio_true,
			}
		else:
			return loss