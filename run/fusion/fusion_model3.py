import torch
from torch import nn
from torch_geometric.nn import Sequential
from sae import AutoEncoderNew
from sae import cross_entropy_loss, correlation
from fusion_gnn3 import EncodeGNN, MergeGNN


class FusionModel(nn.Module):
	def __init__(self, input_dim, embedding_dim=64, autoencoder=AutoEncoderNew, gnn_nlayers=1, max_obj=16, position='rel', **kwargs):
		super().__init__()
		self.gnn_nlayers = gnn_nlayers
		self.max_obj = max_obj
		self.position = position
		self.autoencoder = autoencoder(dim=input_dim, hidden_dim=embedding_dim, max_n=max_obj)
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
		return gnn


	def forward(self, data):
		breakpoint()


	def loss(self):
		autoencoder_loss = self.autoencoder_loss()
		filter_loss = self.filter_loss(return_accuracy=True)
		loss = filter_loss["filter_loss"] + autoencoder_loss["ae_loss"]
		return {
			"loss": loss,
			**autoencoder_loss,
			**filter_loss,
		}