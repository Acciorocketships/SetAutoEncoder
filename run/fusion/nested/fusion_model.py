import torch
from torch import nn
from sae import AutoEncoderNew
from fusion_gnn import Encoder, Decoder, MergeGNN
from sae.util import *


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
		n_agents = shape_nested(obj_idx)
		print("n_agents", n_agents)
		print("max edge_idx", torch.max(edge_idx))

		# Encoder
		encoder_input = {
			"x": x,
			"obj_idx": obj_idx,
			"pos": agent_pos,
		}
		encoder_output = self.encoder(**encoder_input)

		# Merge GNN
		layer_output = {
			"edge_index": edge_idx,
			"pos": agent_pos,
			**encoder_output,
		}
		layer_outputs = []
		for layer in range(self.gnn_nlayers):
			layer_output = self.merge_gnn(**layer_output)
			layer_outputs.append(layer_output)


	def loss(self):
		autoencoder_loss = self.autoencoder_loss()
		filter_loss = self.filter_loss(return_accuracy=True)
		loss = filter_loss["filter_loss"] + autoencoder_loss["ae_loss"]
		return {
			"loss": loss,
			**autoencoder_loss,
			**filter_loss,
		}