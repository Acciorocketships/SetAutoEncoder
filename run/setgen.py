import numpy as np
import torch
from torch import nn
import wandb
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sae import DecoderInner as Decoder
from sae.mlp import build_mlp
from sae import get_loss_idxs, batch_to_set_lens
from sae import min_permutation_loss, fixed_order_loss, cross_entropy_loss


## Run

def experiments():
	trials = {
		"vanilla": {},
	}
	default = {
		"set_size": 7,
		"batch_size": 64,
		"loss_type": "fixed_order",
		"epochs": 10000,
	}
	for name, cfg in trials.items():
		config = default.copy()
		config.update(cfg)
		config["name"] = name
		run(**config)

def run(
			set_size = 8,
			epochs = 10000,
			batch_size = 32,
			name = None,
			loss_type="fixed_order",
		):

	config = locals()

	wandb.init(
			entity = "prorok-lab",
			project = "sae-setgen",
			name = name,
			config = config,
		)

	model = VariableNeuralNetwork(input_dim=set_size, output_dim=set_size, max_n=set_size)

	optim = Adam(model.parameters())

	for t in range(epochs):

		batch = create_dataset(set_size)
		loader = DataLoader(batch.to_data_list(), batch_size=batch_size, shuffle=True)

		for data in loader:
			x = data.y # store y in x so that data.batch matches y
			y = data.x
			yhat, batch_out = model(x)

			n = batch_to_set_lens(data.batch, x.shape[0])
			n_pred = batch_to_set_lens(batch_out, x.shape[0])
			n_pred_logits = model.get_n_pred()
			pred_idx, tgt_idx = get_loss_idxs(n_pred, n)

			# loss
			if loss_type == "fixed_order":
				class_loss = fixed_order_loss(
					yhat=yhat[pred_idx],
					y=y[tgt_idx],
					batch=batch_out[pred_idx],
					loss_fn=cross_entropy_loss,
				)
			elif loss_type == "min_permutation":
				class_loss = min_permutation_loss(
					yhat=yhat[pred_idx],
					y=y[tgt_idx],
					batch=batch_out[pred_idx],
					loss_fn=cross_entropy_loss,
				)
			size_loss = CrossEntropyLoss()(n_pred_logits, n)
			loss = class_loss + size_loss

			wandb.log({
						"class_loss": class_loss,
						"size_loss": size_loss,
						"total_loss": loss,
					})

			loss.backward()
			optim.step()
			optim.zero_grad()

	wandb.finish()


## Dataset

def create_dataset(n):
	datapts = []
	for i in range(2**n):
		x = bin_array(i, n)
		y = torch.nonzero(x)
		datapt = Data(x=y, y=x.unsqueeze(0))
		datapts.append(datapt)
	return Batch.from_data_list(datapts)

def bin_array(num, m):
    return torch.tensor(np.array(list(np.binary_repr(num).zfill(m))[::-1]).astype(np.float32))


## Model

class VariableNeuralNetwork(nn.Module):

	def __init__(self, input_dim, output_dim, max_n=None):
		super().__init__()
		self.input_dim = input_dim
		self.hidden_dim = input_dim
		self.output_dim = output_dim
		self.max_n = max_n
		self.encoder = build_mlp(input_dim=self.input_dim, output_dim=self.hidden_dim, nlayers=3, midmult=1.5, nonlinearity=nn.ReLU)
		self.decoder = Decoder(hidden_dim=self.hidden_dim, dim=self.output_dim, max_n=self.max_n)

	def forward(self, x):
		enc = self.encoder(x)
		return self.decoder(enc)

	def get_n_pred(self):
		return self.decoder.get_n_pred()


if __name__ == '__main__':
	experiments()