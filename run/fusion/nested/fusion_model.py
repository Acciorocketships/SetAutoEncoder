import torch
from torch import nn
from sae import AutoEncoderNew
from sae.loss import cross_entropy_loss
from sae.util import combine_dicts
from fusion_gnn import Encoder, Decoder, MergeGNN


class FusionModel(nn.Module):
	def __init__(self, input_dim, embedding_dim=64, autoencoder=AutoEncoderNew, gnn_nlayers=1, max_obj=16, position='rel', **kwargs):
		super().__init__()
		self.gnn_nlayers = gnn_nlayers
		self.max_obj = max_obj
		self.position = position
		self.autoencoder = autoencoder(dim=input_dim, hidden_dim=embedding_dim, max_n=max_obj)
		self.encoder = Encoder(autoencoder=self.autoencoder, position=self.position, **kwargs)
		self.decoder = Decoder(autoencoder=self.autoencoder, position=self.position, **kwargs)
		self.merge_gnn = MergeGNN(autoencoder=self.autoencoder,position=self.position,**kwargs)


	def forward(self, data):
		# Get Data
		x = data["obj"]
		obj_idx = data["obj_idx"]
		agent_pos = data["agent_pos"]
		edge_idx = data["edge_idx"]

		# Training Data
		self.ae_train_data = []
		self.filter_train_data = []

		# Encoder
		encoder_input = {
			"x": x,
			"obj_idx": obj_idx,
			"pos": agent_pos,
		}
		encoder_output = self.encoder(**encoder_input)

		# Merge GNN
		layer_output = encoder_output
		for layer in range(self.gnn_nlayers):
			layer_input = {
				"edge_index": edge_idx,
				"pos": agent_pos,
				**layer_output,
			}
			layer_output = self.merge_gnn(**layer_input)
			self.ae_train_data.append(self.merge_gnn.get_autoencoder_training_data())
			self.filter_train_data.append(self.merge_gnn.get_filter_training_data())

		decoder_output = self.decoder(**layer_output)
		self.ae_train_data.append(self.decoder.get_autoencoder_training_data())

		return decoder_output


	def loss(self):
		autoencoder_loss = self.autoencoder_loss()
		filter_loss = self.filter_loss()
		loss = autoencoder_loss["loss"]
		# loss = filter_loss["loss"] + autoencoder_loss["loss"]
		autoencoder_loss["autoencoder loss"] = autoencoder_loss["loss"]
		filter_loss["filter loss"] = filter_loss["loss"]
		del autoencoder_loss["loss"]
		del filter_loss["loss"]
		return {
			"loss": loss,
			**autoencoder_loss,
			**filter_loss,
		}


	def autoencoder_loss(self):
		loss_data = []
		for data in self.ae_train_data:
			loss_datai = self.autoencoder.loss(data)
			loss_data.append(loss_datai)
		loss = combine_dicts(loss_data)
		return loss


	def filter_loss(self):
		loss_data = []
		for data in self.filter_train_data:
			classes = data["classes"]
			labels = data["labels"]
			size_loss = torch.mean(cross_entropy_loss(classes, labels))
			pred_class = torch.argmax(classes, dim=-1)
			correct = (labels == pred_class)
			accuracy = torch.sum(correct) / correct.numel()
			pred_ratio = torch.sum(pred_class) / pred_class.numel()
			truth_ratio = torch.sum(labels) / labels.numel()
			loss_datai = {
				"loss": size_loss,
				"filter pred ratio": pred_ratio,
				"filter truth ratio": truth_ratio,
				"filter accuracy": accuracy,
			}
			loss_data.append(loss_datai)
		loss = combine_dicts(loss_data)
		return loss