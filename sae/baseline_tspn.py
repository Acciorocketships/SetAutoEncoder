import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from sae.mlp import build_mlp
from sae.positional import PositionalEncoding
from sae.loss import get_loss_idxs, correlation, min_permutation_loss, mean_squared_loss
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
		x = vars["x"]
		xr = vars["xr"]
		batch = vars["batch"]
		mse_loss = min_permutation_loss(
			yhat=xr[pred_idx],
			y=x[tgt_idx],
			batch=batch[tgt_idx],
			loss_fn=mean_squared_loss,
		)
		crossentropy_loss = CrossEntropyLoss()(vars["n_pred_logits"], vars["n"])
		loss = mse_loss + crossentropy_loss
		corr = correlation(x[tgt_idx], xr[pred_idx])
		return {
			"loss": loss,
			"crossentropy_loss": crossentropy_loss,
			"mse_loss": mse_loss,
			"corr": corr,
		}



class Encoder(nn.Module):

	def __init__(self, dim, hidden_dim=64, max_n=8, **kwargs):
		super().__init__()
		# Params
		self.input_dim = dim
		self.hidden_dim = hidden_dim
		self.max_n = max_n + 1
		self.layernorm = kwargs.get("layernorm_encoder", False)
		self.nonlinearity = kwargs.get("activation_encoder", nn.Tanh)
		# Modules
		self.pos_gen = PositionalEncoding(dim=self.max_n, mode=kwargs.get('pe', 'onehot'))
		self.enc_psi = build_mlp(input_dim=self.input_dim, output_dim=self.hidden_dim, nlayers=2, midmult=1., layernorm=True, nonlinearity=self.nonlinearity)
		self.enc_phi = build_mlp(input_dim=self.hidden_dim+self.max_n, output_dim=self.hidden_dim, nlayers=2, midmult=1., layernorm=self.layernorm, nonlinearity=self.nonlinearity)
		self.rank = torch.nn.Linear(self.input_dim, 1)

	def sort(self, x):
		mag = self.rank(x).reshape(-1)
		_, idx = torch.sort(mag, dim=0)
		return x[idx], idx

	def forward(self, x, batch=None):
		# x: n x input_dim
		_, input_dim = x.shape
		if batch is None:
			batch = torch.zeros(x.shape[0])

		n = scatter(src=torch.ones(x.shape[0]), index=batch, reduce='sum').long()  # batch_size
		self.n = n
		self.x = x
		self.batch = batch

		y1 = self.enc_psi(x) # total_nodes x hidden_dim

		y2 = scatter(src=y1, index=batch, dim=-2, reduce='sum') # batch_size x dim

		pos_n = self.pos_gen(n) # batch_size x max_n
		y3 = torch.cat([y2, pos_n], dim=-1) # batch_size x (hidden_dim + max_n)

		z = self.enc_phi(y3) # batch_size x hidden_dim
		return z

	def get_x_perm(self):
		'Returns: the permutation applied to the inputs (shape: ninputs)'
		return torch.arange(self.x.shape[0])

	def get_z(self):
		'Returns: the latent state (shape: batch x hidden_dim)'
		return self.z

	def get_batch(self):
		'Returns: the batch idxs of the inputs (shape: ninputs)'
		return self.batch

	def get_x(self):
		'Returns: the sorted inputs, x[x_perm] (shape: ninputs x d)'
		return self.x

	def get_n(self):
		'Returns: the number of elements per batch (shape: batch)'
		return self.n


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
		ptr = torch.cat([torch.zeros(1), torch.cumsum(n, dim=0)], dim=0).int()
		max_n = torch.max(n)
		x0z = torch.zeros(z.shape[0], max_n, self.output_dim + self.hidden_dim)
		mask = torch.zeros(z.shape[0], max_n).bool()
		for i in range(n.shape[0]):
			x0z[i, :n[i], :self.output_dim] = x0[ptr[i]:ptr[i + 1], :]
			x0z[i, :n[i], self.output_dim:] = rep_z[ptr[i]:ptr[i + 1], :]
			mask[i,:n[i]] = True

		xz_padded = self.transformer(src=x0z, src_key_padding_mask=~mask)

		x_padded = self.conv(xz_padded.permute(0,2,1)).permute(0,2,1)
		x_padded_flat = x_padded.reshape(z.shape[0] * max_n, self.output_dim)
		mask_flat = mask.view(z.shape[0] * max_n)
		x = x_padded_flat[mask_flat,:]

		batch = torch.repeat_interleave(torch.arange(n.shape[0]), n, dim=0)
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

	enc = Encoder(dim=dim)
	dec = Decoder(dim=dim)

	data_list = []
	for i in range(batch_size):
		n = torch.randint(low=1, high=max_n, size=(1,))
		x = torch.randn(n[0], dim)
		d = Data(x=x, y=x)
		data_list.append(d)
	data = Batch.from_data_list(data_list)

	z = enc(data.x, data.batch)
	xr, batchr = dec(z)

	print(data.x.shape, xr.shape)
	print(data.batch.shape, batchr.shape)
