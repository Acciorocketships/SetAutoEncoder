import torch
from torch import nn
from sae.util import scatter
from sae.mlp import build_mlp
from sae.positional import PositionalEncoding
from sae.loss import get_loss_idxs, correlation, mean_squared_loss
from sae.sae_model import AutoEncoder as BaseAutoEncoder
from torch_scatter.composite import scatter_softmax

# sae_new2: new architecture where keys and values are encoded and passed separately

class AutoEncoder(nn.Module):

	def __init__(self, dim=1, hidden_dim=64, **kwargs):
		'''
		Must have self.encoder and self.decoder objects, which follow the encoder and decoder interfaces
		'''
		super().__init__()
		key_dim = 4
		key_storage_size = 64
		self.key_storage = BaseAutoEncoder(dim=key_dim, hidden_dim=key_storage_size)
		self.encoder = Encoder(dim=dim, hidden_dim=hidden_dim, key_storage=self.key_storage, **kwargs)
		self.decoder = Decoder(dim=dim, hidden_dim=hidden_dim, key_storage=self.key_storage, **kwargs)
		# self.decoder.key_net = self.encoder.key_net

	def forward(self, x, batch=None):
		z, keyenc = self.encoder(x, batch)
		xr, batchr = self.decoder(z, keyenc)
		return xr, batchr

	def get_vars(self):
		self.vars = {
			"n_pred": self.decoder.get_n_pred(),
			"n": self.encoder.get_n(),
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
		mse_loss = torch.mean(mean_squared_loss(x[tgt_idx], xr[pred_idx]))
		if torch.isnan(mse_loss):
			mse_loss = 0
		loss = 100 * mse_loss
		corr = correlation(x[tgt_idx], xr[pred_idx])
		key_storage_loss = self.key_storage.loss()
		return {
			"loss": loss + key_storage_loss["loss"],
			"mse_loss": mse_loss,
			"corr": corr,
			"key_storage_size_loss": key_storage_loss["size_loss"],
			"key_storage_mse_loss": key_storage_loss["mse_loss"],
			"key_storage_corr": key_storage_loss["corr"],
		}



class Encoder(nn.Module):

	def __init__(self, dim, hidden_dim, key_storage, **kwargs):
		super().__init__()
		# Params
		self.input_dim = dim
		self.hidden_dim = hidden_dim
		self.key_storage = key_storage
		# Modules
		self.key_net = build_mlp(input_dim=key_storage.encoder.input_dim, output_dim=self.hidden_dim, nlayers=2, midmult=1.,layernorm=True, nonlinearity=nn.Mish)
		self.val_net = build_mlp(input_dim=self.input_dim, output_dim=self.hidden_dim, nlayers=2, midmult=1.,layernorm=True, nonlinearity=nn.Mish)
		self.mapping = build_mlp(input_dim=self.input_dim, output_dim=key_storage.encoder.input_dim, nlayers=1, midmult=1.,layernorm=False, nonlinearity=nn.Mish)

	def forward(self, x, batch=None, n_batches=None):
		# x: n x input_dim
		if batch is None:
			batch = torch.zeros(x.shape[0], device=x.device)

		# Keys
		key0 = self.mapping(x)
		keys = scatter_softmax(key0, batch, dim=-2)
		# self.pos_gen = PositionalEncoding(dim=self.key_storage.encoder.input_dim, mode="onehot")
		# n = scatter(src=torch.ones(x.shape[0], device=x.device), index=batch, dim_size=n_batches).long()
		# k = torch.cat([torch.arange(ni, device=x.device) for ni in n], dim=0).int() # batch_size]
		# keys = self.pos_gen(k) # batch_size x hidden_dim

		# Encoder
		y = self.val_net(x) * self.key_net(keys)  # total_nodes x hidden_dim
		z = scatter(src=y, index=batch, dim=-2, dim_size=n_batches)  # batch_size x dim

		# Store Keys
		keyenc = self.key_storage.encoder(keys, batch)

		self.x = x
		self.z = z
		self.batch = batch
		return z, keyenc

	def get_x_perm(self):
		'Returns: the permutation applied to the inputs (shape: ninputs)'
		return torch.arange(x.shape[0])
	
	def get_z(self):
		'Returns: the latent state (shape: batch x hidden_dim)'
		return self.z

	def get_batch(self):
		'Returns: the batch idxs of the inputs (shape: ninputs)'
		return self.batch

	def get_x(self):
		'Returns: the sorted inputs, x[x_perm] (shape: ninputs x dim)'
		return self.x
		# return self.x[self.get_x_perm()]

	def get_n(self):
		'Returns: the number of elements per batch (shape: batch)'
		return self.key_storage.encoder.get_n()



class Decoder(nn.Module):

	def __init__(self, dim, hidden_dim, key_storage, **kwargs):
		super().__init__()
		# Params
		self.output_dim = dim
		self.hidden_dim = hidden_dim
		self.key_storage = key_storage
		# Modules
		self.key_net = build_mlp(input_dim=key_storage.encoder.input_dim, output_dim=self.hidden_dim, nlayers=2, midmult=1., layernorm=True, nonlinearity=nn.Mish)
		self.decoder = build_mlp(input_dim=self.hidden_dim, output_dim=self.output_dim, nlayers=2, midmult=1., layernorm=False, nonlinearity=nn.Mish)

	def forward(self, z, keyenc):
		keys, batch = self.key_storage.decoder(keyenc)
		n_pred = self.key_storage.decoder.get_n_pred()

		q = self.key_net(keys)

		vals_rep = torch.repeat_interleave(z, n_pred, dim=0)
		zp = vals_rep * q

		x = self.decoder(zp)

		self.x = x
		self.batch = batch
		self.n_pred = n_pred
		return x, batch

	def get_batch_pred(self):
		'Returns: the batch idxs of the outputs x (shape: noutputs)'
		return self.batch

	def get_x_pred(self):
		'Returns: the outputs x (shape: noutputs x d)'
		return self.x

	def get_n_pred(self):
		'Returns: the predicted cardinality (shape: batch)'
		return self.n_pred



if __name__ == '__main__':

	dim = 3
	max_n = 5
	batch_size = 16

	enc = Encoder(dim=dim)
	dec = Decoder(dim=dim)

	data_list = []
	batch_list = []
	for i in range(batch_size):
		n = torch.randint(low=1, high=max_n, size=(1,))
		x = torch.randn(n[0], dim)
		data_list.append(x)
		batch_list.append(torch.ones(n) * i)
	data = torch.cat(data_list, dim=0)
	batch = torch.cat(batch_list, dim=0).int()

	z = enc(data, batch)
	xr, batchr = dec(z)

	print(x.shape, xr.shape)
	print(batch.shape, batchr.shape)

