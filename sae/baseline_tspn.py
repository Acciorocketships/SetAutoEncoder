import torch
from torch import nn
from sae.mlp import build_mlp
from sae.baseline_dspn import Encoder
from sae.loss import get_loss_idxs, correlation, min_permutation_idxs, mean_squared_loss
from torch.nn import CrossEntropyLoss


class AutoEncoder(nn.Module):

	def __init__(self, *args, **kwargs):
		'''
		Must have self.encoder and self.decoder objects, which follow the encoder and decoder interfaces
		'''
		super().__init__()
		self.encoder = Encoder(*args, **kwargs)
		self.decoder = Decoder(*args, **kwargs)

	def forward(self, x, batch=None):
		z = self.encoder(x, batch)
		xr, batchr = self.decoder(z)
		return xr, batchr

	def get_vars(self):
		self.vars = {
			"n_pred_logits": self.decoder.get_n_pred_logits(),
			"n_pred": self.decoder.get_n_pred(),
			"n": self.encoder.get_n(),
			"batch": self.encoder.get_batch(),
			"x": self.encoder.get_x(),
			"xr": self.decoder.get_x_pred(),
		}
		return self.vars

	def loss(self, vars=None):
		'''
		Input: the output of self.get_vars()
		Returns: a dict of info which must include 'loss'
		'''
		if vars is None:
			vars = self.get_vars()
		pred_idx, tgt_idx = get_loss_idxs(vars["n_pred"], vars["n"])
		x = vars["x"][tgt_idx]
		xr = vars["xr"][pred_idx]
		batch = vars["batch"][tgt_idx]
		perm = min_permutation_idxs(
			yhat=xr,
			y=x,
			batch=batch,
			loss_fn=mean_squared_loss,
		)
		xr = xr[perm]
		mse_loss = torch.mean(mean_squared_loss(x, xr))
		crossentropy_loss = CrossEntropyLoss()(vars["n_pred_logits"], vars["n"])
		loss = mse_loss + crossentropy_loss
		corr = correlation(x, xr)
		return {
			"loss": loss,
			"crossentropy_loss": crossentropy_loss,
			"mse_loss": mse_loss,
			"corr": corr,
		}


class Decoder(nn.Module):

	def __init__(self, dim, hidden_dim=64, max_n=8, **kwargs):
		super().__init__()
		# Params
		self.output_dim = dim
		self.hidden_dim = hidden_dim
		self.max_n = max_n + 1
		# Modules
		self.size_pred = build_mlp(input_dim=self.hidden_dim, output_dim=self.max_n)
		self.init_mu = torch.nn.parameter.Parameter(torch.randn(dim))
		self.init_sigma = torch.nn.parameter.Parameter(torch.rand(dim))
		self.transformer = torch.nn.TransformerEncoder(
			encoder_layer=torch.nn.TransformerEncoderLayer(
				d_model=dim+self.hidden_dim,
				nhead=2,
				dim_feedforward=self.hidden_dim,
				batch_first=True,
			),
			num_layers=3,
			enable_nested_tensor=True,
		)
		self.conv = torch.nn.Conv1d(
			in_channels=self.hidden_dim+self.output_dim,
			out_channels=self.output_dim,
			kernel_size=1,
		)

	def forward(self, z):
		# z: batch_size x hidden_dim
		n_pred = self.size_pred(z)  # batch_size x max_n
		n = torch.argmax(n_pred, dim=-1)
		self.n_pred_logits = n_pred
		self.n_pred = n

		num_outputs = torch.sum(n)

		sample_dist = torch.distributions.Normal(loc=self.init_mu, scale=torch.abs(self.init_sigma))
		x0 = sample_dist.rsample(sample_shape=(num_outputs,))
		rep_z = torch.repeat_interleave(z, n, dim=0)
		ptr = torch.cat([torch.zeros(1, device=z.device), torch.cumsum(n, dim=0)], dim=0).int()
		max_n = torch.max(n)
		x0z = torch.zeros(z.shape[0], max_n, self.output_dim + self.hidden_dim, device=z.device)
		mask = torch.zeros(z.shape[0], max_n, device=z.device).bool()
		for i in range(n.shape[0]):
			x0z[i, :n[i], :self.output_dim] = x0[ptr[i]:ptr[i + 1], :]
			x0z[i, :n[i], self.output_dim:] = rep_z[ptr[i]:ptr[i + 1], :]
			mask[i,:n[i]] = True

		xz_padded = self.transformer(src=x0z, src_key_padding_mask=~mask)

		x_padded = self.conv(xz_padded.permute(0,2,1)).permute(0,2,1)
		x_padded_flat = x_padded.reshape(z.shape[0] * max_n, self.output_dim)
		mask_flat = mask.view(z.shape[0] * max_n)
		x = x_padded_flat[mask_flat,:]

		batch = torch.repeat_interleave(torch.arange(n.shape[0], device=z.device), n, dim=0)
		self.batch = batch
		self.x = x
		return x, batch

	def get_batch_pred(self):
		'Returns: the batch idxs of the outputs x (shape: noutputs)'
		return self.batch

	def get_x_pred(self):
		'Returns: the outputs x (shape: noutputs x d)'
		return self.x

	def get_n_pred_logits(self):
		'Returns: the class logits for each possible n, up to max_n (shape: batch x max_n)'
		return self.n_pred_logits

	def get_n_pred(self):
		'Returns: the actual n, obtained by taking the argmax over n_pred_logits (shape: batch)'
		return self.n_pred



if __name__ == '__main__':

	dim = 4
	max_n = 5
	batch_size = 16

	# enc = Encoder(dim=dim)
	# dec = Decoder(dim=dim)
	ae = AutoEncoder(dim=dim, max_n=max_n)

	data_list = []
	batch_list = []
	for i in range(batch_size):
		n = torch.randint(low=1, high=max_n, size=(1,))
		x = torch.randn(n[0], dim)
		data_list.append(x)
		batch_list.append(torch.ones(n) * i)
	x = torch.cat(data_list, dim=0)
	batch = torch.cat(batch_list, dim=0).int()

	xr, batchr = ae(x, batch)

	print(x.shape, xr.shape)
	print(batch.shape, batchr.shape)

	loss = ae.loss()

	breakpoint()

