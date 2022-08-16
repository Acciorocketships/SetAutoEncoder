import torch
from torch import nn
from torch_geometric.nn import Sequential
from sae import get_loss_idxs, batch_to_set_lens
from sae import cross_entropy_loss, mean_squared_loss, correlation, min_permutation_idxs
from fusion_gnn import EncodeGNN, MergeGNN


class FusionModel(nn.Module):
	def __init__(self, input_dim, embedding_dim=256, gnn_nlayers=1, max_agents=16, max_obj=16, position='rel', **kwargs):
		super().__init__()
		self.input_dim = input_dim
		self.embedding_dim = embedding_dim
		self.pos_dim = 2 if (position is not None) else 0
		self.gnn_nlayers = gnn_nlayers
		self.max_obj = max_obj
		self.max_agents = max_agents
		self.position = position
		self.encode_gnn = EncodeGNN(in_channels=self.input_dim+self.pos_dim, out_channels=self.embedding_dim, max_obj=self.max_obj, position=self.position, **kwargs)
		self.merge_gnn = self.create_merge_gnn(**kwargs)
		self.decoder = self.merge_gnn[0].input_decoder
		self.encode_gnn.encoder = self.merge_gnn[0].merge_encoder


	def create_merge_gnn(self, **kwargs):
		layers = []
		signatures = []
		merge_gnn_layer = MergeGNN(
			in_channels=self.embedding_dim,
			out_channels=self.embedding_dim,
			orig_dim=self.input_dim+self.pos_dim,
			max_agents=self.max_agents,
			max_obj=self.max_obj,
			position=self.position,
			**kwargs,
		)
		for i in range(self.gnn_nlayers):
			layers.append(merge_gnn_layer)
			signatures.append("x, edge_index, pos -> x")
		gnn = Sequential("x, edge_index, pos", zip(layers, signatures))
		gnn.reset_values = gnn[0].reset_values
		gnn.get_values = gnn[0].get_values
		return gnn


	def forward(self, data):
		# self.forward_true(data)

		obj_x = data['object'].x
		obj_pos = data['object'].pos
		agent_pos = data['agent'].pos
		obj_agent_edge_index = data[('agent', 'observe', 'object')].edge_index # [agent_idx, obj_idx]
		agent_edge_index = data[('agent', 'communicate', 'agent')].edge_index
		obj_per_agent_obs = batch_to_set_lens(obj_agent_edge_index[0, :], batch_size=data['agent'].pos.shape[0])

		self.merge_gnn.reset_values()

		self.enc = self.encode_gnn(x=obj_x, edge_index=obj_agent_edge_index, posx=obj_pos, posa=agent_pos)

		obj_idx = obj_agent_edge_index[1, self.encode_gnn.x_perm]
		agent_idx = obj_agent_edge_index[0, self.encode_gnn.x_perm]
		n_agents = obj_x.shape[0]
		obj_idx_nested = torch.nested_tensor([obj_idx[agent_idx==i] for i in range(n_agents)])
		# x_nested = torch.nested_tensor([self.encode_gnn.input[agent_idx==i] for i in range(n_agents)])

		self.merge_gnn[0].values["obj_idx"].append(obj_idx)
		self.merge_gnn[0].values["batch_output"].append(agent_idx)
		self.merge_gnn[0].values["n_output"].append(obj_per_agent_obs)
		self.merge_gnn[0].values["x_output"].append(self.encode_gnn.input)
		self.merge_gnn[0].values["perm_output"].append(self.encode_gnn.x_perm)

		self.merged = self.merge_gnn(x=self.enc, edge_index=agent_edge_index, pos=agent_pos, obj_idx=obj_idx_nested)

		self.decoded, self.batch = self.decoder(self.merged)

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


	def loss(self, data):
		# Merge Layers Loss
		merge_size_loss = self.merge_size_loss()
		merge_element_loss, merge_corr = self.merge_corr_loss(return_corr=True)
		decoder_size_loss = self.decoder_size_loss()
		decoder_element_loss, decoder_corr = self.decoder_corr_loss(return_corr=True)
		# filter_data = self.filter_loss(return_accuracy=True, num_layers=2)
		filter_loss = 0 # filter_data["loss"]

		decoder_loss = 0.1 * (decoder_size_loss + 100 * decoder_element_loss)
		merge_loss = merge_size_loss + 100 * merge_element_loss
		loss = 1. * merge_loss + 0.3 * filter_loss + 0 * decoder_loss

		# del filter_data["loss"]
		return {
			"loss": loss,
			"merge_element_loss": merge_element_loss / self.gnn_nlayers,
			"merge_size_loss": merge_size_loss,
			"merge_corr": merge_corr,
			"decoder_size_loss": decoder_size_loss,
			"decoder_element_loss": decoder_element_loss,
			"decoder_corr": decoder_corr,
			"filter_loss": filter_loss,
			# **filter_data,
			# **self.stats(data),
		}


	def merge_size_loss(self):
		layer_size_losses = []
		n_trues = self.merge_gnn.get_values("n_output")
		n_pred_logits = self.merge_gnn.get_values("n_pred_logits")
		max_n = self.merge_gnn.get_values("max_n")
		for i in range(self.gnn_nlayers):
			crossentropy = self.size_loss(n_true=n_trues[i], n_pred_logits=n_pred_logits[i], max_n=max_n)
			layer_size_losses.append(crossentropy)
		return sum(layer_size_losses)


	def merge_corr_loss(self, return_corr=False):
		layer_losses = []
		layer_corrs = []
		n_trues = self.merge_gnn.get_values("n_output")
		x_trues = self.merge_gnn.get_values("x_output")
		perms = self.merge_gnn.get_values("perm_output")
		n_preds = self.merge_gnn.get_values("n_pred")
		x_preds = self.merge_gnn.get_values("x_pred")
		batch_preds = self.merge_gnn.get_values("batch_pred")
		for i in range(self.gnn_nlayers):
			mse = self.corr_loss(
				n_true=n_trues[i],
				n_pred=n_preds[i],
				x_true=x_trues[i],
				x_pred=x_preds[i],
				batch_pred=batch_preds[i],
				perm=perms[i],
				return_corr=return_corr,
			)
			if return_corr:
				mse, corr = mse
				layer_corrs.append(corr)
			layer_losses.append(mse)
		loss = sum(layer_losses)
		corr = torch.mean(torch.tensor(layer_corrs))
		if return_corr:
			return loss, corr
		else:
			return loss

	def decoder_size_loss(self):
		return self.size_loss(
			n_true=self.merge_gnn.get_values("n_output")[-1],
			n_pred_logits=self.decoder.get_n_pred_logits(),
			max_n=self.decoder.max_n
		)

	def decoder_corr_loss(self, return_corr=False):
		return self.corr_loss(
			n_true=self.merge_gnn.get_values("n_output")[-1],
			n_pred=self.decoder.get_n_pred(),
			x_true=self.merge_gnn.get_values("x_output")[-1],
			x_pred=self.decoded,
			batch_pred=self.batch,
			perm=self.merge_gnn.get_values("perm_output")[-1],
			return_corr=return_corr,
		)


	def size_loss(self, n_true, n_pred_logits, max_n):
		n_true = torch.minimum(n_true, torch.tensor(max_n-1))
		crossentropy = torch.mean(cross_entropy_loss(n_pred_logits, n_true))
		return crossentropy


	def corr_loss(self, n_true, n_pred, x_true, x_pred, batch_pred, perm=None, return_corr=False):
		pred_idx, true_idx = get_loss_idxs(n_pred, n_true)
		if perm is None:
			x_pred_subset = x_pred[pred_idx]
			x_true_subset = x_true[true_idx]
			batch_subset = batch_pred[pred_idx]
			idxs = min_permutation_idxs(
				yhat=x_pred_subset,
				y=x_true_subset,
				batch=batch_subset,
				loss_fn=mean_squared_loss,
			)
			x_pred_subset = x_pred_subset[idxs]
		else:
			x_true = x_true[perm]
			x_pred_subset = x_pred[pred_idx]
			x_true_subset = x_true[true_idx]
		mses = mean_squared_loss(x_pred_subset, x_true_subset.detach())
		mse = torch.mean(mses)
		if return_corr:
			corr = correlation(x_pred_subset, x_true_subset)
			return mse, corr
		else:
			return mse


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
				"loss": loss,
				"filter_acc": acc,
				"corr_same": corr_same,
				"corr_diff": corr_diff,
				"ratio_pred": ratio_pred,
				"ratio_true": ratio_true,
			}
		else:
			return loss


	def stats(self, data):
		def get_batch(data):
			num_agents = data['agent'].batch.shape[0]
			data_list = data.to_data_list()
			agents_per_batch = torch.tensor([d['agent'].pos.shape[0] for d in data_list])
			objects_per_batch = torch.tensor([d['object'].pos.shape[0] for d in data_list])
			obj_x_all = torch.cat([d['object'].x.repeat(n, 1) for n, d in zip(agents_per_batch, data_list)])
			obj_pos_all = torch.cat([d['object'].pos.repeat(n, 1) for n, d in zip(agents_per_batch, data_list)])
			agent_pos_all = torch.cat(
				[d['agent'].pos.repeat_interleave(n, dim=0) for n, d in zip(objects_per_batch, data_list)])
			num_objects_per_agent = objects_per_batch.repeat_interleave(agents_per_batch)
			obj_batch_all = torch.arange(num_agents).repeat_interleave(num_objects_per_agent)
			return {
				"batch": obj_batch_all,
				"obj_per_agent": num_objects_per_agent,
				"obj_x": obj_x_all,
				"obj_pos": obj_pos_all,
				"agent_pos": agent_pos_all,
			}
		truth_data = get_batch(data)
		y = self.encode_gnn.message(truth_data['obj_x'], truth_data['obj_pos'], truth_data['agent_pos'])
		y_batch = truth_data["batch"]
		y_n = truth_data["obj_per_agent"]
		yhat = self.decoded
		yhat_n_logits = self.decoder.get_n_pred_logits()
		yhat_n = self.decoder.get_n_pred()
		pred_idx, tgt_idx = get_loss_idxs(yhat_n, y_n)
		y_match = y[tgt_idx]
		yhat_match = yhat[pred_idx]
		batch = y_batch[tgt_idx]
		perm = min_permutation_idxs(yhat=yhat_match, y=y_match, batch=batch, loss_fn=mean_squared_loss)
		y_ord = y_match
		yhat_ord = yhat_match[perm]
		element_error = torch.mean(mean_squared_loss(yhat_ord, y_ord))
		size_error = torch.mean(cross_entropy_loss(yhat_n_logits, y_n))
		size_accuracy = torch.mean((yhat_n == y_n).float())
		corr = correlation(y_ord, yhat_ord)
		return {
			"element_error": element_error,
			"size_error": size_error,
			"size_accuracy": size_accuracy,
			"corr": corr,
		}