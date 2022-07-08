import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing, Sequential
from torch import Tensor
from typing import Optional
from sae import EncoderNew, DecoderNew
from sae import get_loss_idxs, batch_to_set_lens
from sae import cross_entropy_loss, mean_squared_loss, correlation, min_permutation_idxs


class FusionModel(nn.Module):
	def __init__(self, input_dim, embedding_dim=128, gnn_nlayers=1, max_agents=16, max_obj=16, position='rel', **kwargs):
		super().__init__()
		self.input_dim = input_dim
		self.embedding_dim = embedding_dim
		self.pos_dim = 2 if (position is not None) else 0
		self.gnn_nlayers = gnn_nlayers
		self.max_obj = max_obj
		self.max_agents = max_agents
		self.position = position
		self.encode_gnn = EncodeGNN(in_channels=self.input_dim, out_channels=self.embedding_dim, max_obj=self.max_obj, position=self.position, **kwargs)
		self.merge_gnn = self.create_merge_gnn(**kwargs)
		self.decoder = DecoderNew(hidden_dim=self.embedding_dim, dim=self.input_dim+self.pos_dim, max_n=self.max_obj, **kwargs)

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
			layers.append(merge_gnn_layer) # same layer repeated
			signatures.append("x, edge_index, pos -> x")
			# layers.append(nn.ReLU)
		gnn = Sequential("x, edge_index, pos", zip(layers, signatures))
		gnn.reset_values = gnn[0].reset_values
		gnn.get_values = gnn[0].get_values
		return gnn

	def forward(self, data):
		obj_x = data['object'].x
		obj_pos = data['object'].pos
		agent_pos = data['agent'].pos
		obj_agent_edge_index = data[('agent', 'observe', 'object')].edge_index
		agent_edge_index = data[('agent', 'communicate', 'agent')].edge_index

		self.merge_gnn.reset_values()
		self.enc = self.encode_gnn(x=obj_x, edge_index=obj_agent_edge_index, posx=obj_pos, posa=agent_pos)
		self.merged = self.merge_gnn(x=self.enc, edge_index=agent_edge_index, pos=agent_pos)
		self.decoded, self.batch = self.decoder(self.merged)

		return self.decoded, self.batch

	def get_batch(self, data):
		num_agents = data['agent'].batch.shape[0]
		data_list = data.to_data_list()
		agents_per_batch = torch.tensor([d['agent'].pos.shape[0] for d in data_list])
		objects_per_batch = torch.tensor([d['object'].pos.shape[0] for d in data_list])
		obj_x_all = torch.cat([d['object'].x.repeat(n,1) for n, d in zip(agents_per_batch, data_list)])
		obj_pos_all = torch.cat([d['object'].pos.repeat(n, 1) for n, d in zip(agents_per_batch, data_list)])
		agent_pos_all = torch.cat([d['agent'].pos.repeat_interleave(n, dim=0) for n, d in zip(objects_per_batch, data_list)])
		num_objects_per_agent = objects_per_batch.repeat_interleave(agents_per_batch)
		obj_batch_all = torch.arange(num_agents).repeat_interleave(num_objects_per_agent)
		return {
			"batch": obj_batch_all,
			"obj_per_agent": num_objects_per_agent,
			"obj_x": obj_x_all,
			"obj_pos": obj_pos_all,
			"agent_pos": agent_pos_all,
		}

	# def get_objs_per_agent(self, data):
	# 	obj_agent_edge = data[('agent', 'observe', 'object')].edge_index
	# 	obj_idxs = obj_agent_edge[1,self.encode_gnn.encoder.get_x_perm()]
	# 	agent_idxs = obj_agent_edge[0,:]
	#
	# 	breakpoint()
	# merge_gnn already maps reconstructed objs to agents
	# compare that to the true objects


	def loss(self, data):
		# Get Data
		obj_per_agent_obs = batch_to_set_lens(
			data[('agent', 'observe', 'object')].edge_index[0, :],
			batch_size=data['agent'].pos.shape[0],
		)

		# Merge Layers Loss
		merge_size_loss = self.merge_size_loss(n_true=obj_per_agent_obs)
		merge_element_loss, merge_corr = self.merge_corr_loss(
			n_true=obj_per_agent_obs,
			x_true=self.encode_gnn.input,
			perm=self.encode_gnn.encoder.get_x_perm(),
			return_corr=True,
		)

		# Decoder Layer Loss
		decoder_size_loss = self.size_loss(
			n_true=self.merge_gnn.get_values("n_output")[-1],
			n_pred_logits=self.decoder.get_n_pred_logits(),
			max_n=self.decoder.max_n
		)
		decoder_element_loss, decoder_corr = self.corr_loss(
			n_true=self.merge_gnn.get_values("n_output")[-1],
			n_pred=self.decoder.get_n_pred(),
			x_true=self.merge_gnn.get_values("x_output")[-1],
			x_pred=self.decoded,
			batch_pred=self.batch,
			perm=self.merge_gnn.get_values("perm_output")[-1],
			return_corr=True,
		)

		loss = merge_size_loss + 10 * merge_element_loss + decoder_size_loss + decoder_element_loss

		return {
			"loss": loss,
			"merge_element_loss": merge_element_loss / self.gnn_nlayers,
			"merge_size_loss": merge_size_loss,
			"decoder_element_loss": decoder_element_loss,
			"decoder_size_loss": decoder_size_loss,
			"merge_corr": merge_corr,
			"decoder_corr": decoder_corr,
			**self.stats(data),
		}

	def merge_size_loss(self, n_true):
		layer_size_losses = []
		n_trues = [n_true] + self.merge_gnn.get_values("n_output")[:-1]
		n_pred_logits = self.merge_gnn.get_values("n_pred_logits")
		max_n = self.merge_gnn.get_values("max_n")
		for i in range(self.gnn_nlayers):
			crossentropy = self.size_loss(n_true=n_trues[i], n_pred_logits=n_pred_logits[i], max_n=max_n)
			layer_size_losses.append(crossentropy)
		return sum(layer_size_losses)

	def merge_corr_loss(self, n_true, x_true, perm, return_corr=False):
		layer_losses = []
		layer_corrs = []
		n_trues = [n_true] + self.merge_gnn.get_values("n_output")[:-1]
		x_trues = [x_true] + self.merge_gnn.get_values("x_output")[:-1]
		perms = [perm] + self.merge_gnn.get_values("perm_output")[:-1]
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
		mses = mean_squared_loss(x_pred_subset, x_true_subset)
		mse = torch.mean(mses)
		if return_corr:
			corr = correlation(x_pred_subset, x_true_subset)
			return mse, corr
		else:
			return mse


	def stats(self, data):
		truth_data = self.get_batch(data)
		y = self.encode_gnn.message(truth_data['obj_x'], truth_data['obj_pos'], truth_data['agent_pos'])
		y_batch = truth_data["batch"]
		y_n = truth_data["obj_per_agent"]
		yhat = self.decoded
		yhat_n_logits = self.decoder.get_n_pred_logits()
		yhat_n = self.decoder.get_n_pred()
		pred_idx, tgt_idx = get_loss_idxs(yhat_n, y_n)
		y_match = y[tgt_idx][:,:4]
		yhat_match = yhat[pred_idx][:,:4]
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



class EncodeGNN(MessagePassing):
	def __init__(self, in_channels, out_channels, max_obj=8, position='abs', **kwargs):
		super().__init__()
		self.input_dim = in_channels
		self.output_dim = out_channels
		self.pos_dim = 2 if position is not None else 0
		self.position = position
		self.encoder = EncoderNew(dim=self.input_dim+self.pos_dim, hidden_dim=self.output_dim, max_n=max_obj)

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
		self.input = inputs
		self.batch = index
		return self.encoder(inputs, batch=index, n_batches=dim_size)


class MergeGNN(MessagePassing):
	def __init__(self, in_channels, out_channels, orig_dim, max_obj=16, position='abs', **kwargs):
		super().__init__()
		self.input_dim = in_channels
		self.output_dim = out_channels
		self.pos_dim = 2 if position is not None else 0
		self.orig_dim = orig_dim
		self.position = position
		self.filter_dist_thres = 0.02
		self.input_decoder = DecoderNew(hidden_dim=self.input_dim, dim=self.orig_dim, max_n=max_obj)
		self.merge_encoder = EncoderNew(dim=self.orig_dim, hidden_dim=self.output_dim, max_n=max_obj)

	def reset_values(self):
		self.values = {
			"n_pred_logits": [],
			"n_pred": [],
			"x_pred": [],
			"batch_pred": [],
			"n_output": [],
			"x_output": [],
			"perm_output": [],
			"max_n": self.input_decoder.max_n
		}

	def get_values(self, key):
		return self.values[key]

	def forward(self, x: Tensor, edge_index: Tensor,  pos: Tensor):
		self.set_decoder_preds(x)
		return self.propagate(x=x, edge_index=edge_index, pos=pos, size=(x.shape[0], x.shape[0]))

	def set_decoder_preds(self, x: Tensor):
		decoded, decoded_batch = self.input_decoder(x)
		self.values["n_pred_logits"].append(self.input_decoder.get_n_pred_logits())
		self.values["n_pred"].append(self.input_decoder.get_n_pred())
		self.values["x_pred"].append(decoded)
		self.values["batch_pred"].append(decoded_batch)

	def message(self, x_j: Tensor, pos_j: Tensor, pos_i: Tensor) -> Tensor:
		decoded, decoded_batch = self.input_decoder(x_j)
		if self.position == 'rel':
			decoded = self.update_rel_pos(decoded, decoded_batch, pos_j, pos_i)
		return (decoded, decoded_batch)

	def update_rel_pos(self, decoded: Tensor, decoded_batch: Tensor, pos_j: Tensor, pos_i: Tensor) -> Tensor:
		decoded_pos = decoded[:,-self.pos_dim:]
		objs_per_agent = batch_to_set_lens(decoded_batch, pos_j.shape[0])
		relposij = torch.repeat_interleave((pos_j - pos_i), objs_per_agent, dim=0)
		new_pos = decoded_pos + relposij
		decoded[:,-self.pos_dim:] = new_pos
		return decoded

	def filter_duplicates(self, x: Tensor, index: Tensor, dim_size: Optional[int] = None, pos_only: bool = True):
		if pos_only:
			feat = x[:, -self.pos_dim:]
		else:
			feat = x
		mask = torch.zeros(x.shape[0], dtype=bool)
		for i in range(dim_size):
			agent_obj_feat = feat[index == i,:]
			obj_pairwise_disp = (agent_obj_feat.unsqueeze(0) - agent_obj_feat.unsqueeze(1)).norm(dim=-1)
			duplicates = torch.triu(obj_pairwise_disp < self.filter_dist_thres, diagonal=1)
			maski = duplicates.sum(dim=0) == 0
			if maski.sum() >= self.merge_encoder.max_n:
				idxs = torch.where(maski)[0][self.merge_encoder.max_n-1:]
				maski[idxs] = False
			mask[index == i] = maski
		return x[mask,:], index[mask]


	def aggregate(self, inputs: Tensor, index: Tensor,
				  ptr: Optional[Tensor] = None,
				  dim_size: Optional[int] = None) -> Tensor:
		x = inputs[0]
		obj_index = index[inputs[1]]
		x, obj_index = self.filter_duplicates(x, obj_index, dim_size=dim_size)
		output = self.merge_encoder(x, batch=obj_index, n_batches=dim_size)
		self.values["n_output"].append(scatter(src=torch.ones(obj_index.shape[0]), index=obj_index, dim_size=dim_size, reduce='sum').long())
		self.values["x_output"].append(x)
		self.values["perm_output"].append(self.merge_encoder.get_x_perm())
		return output