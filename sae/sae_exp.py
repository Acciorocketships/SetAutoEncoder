import torch
from torch import nn
from sae.util import scatter
from sae.mlp import build_mlp
from sae.positional import PositionalEncoding
from sae.loss import get_loss_idxs, correlation, mean_squared_loss


class AutoEncoder(nn.Module):

	def __init__(self, *args, **kwargs):
		'''
		Must have self.encoder and self.decoder objects, which follow the encoder and decoder interfaces
		'''
		super().__init__()
		self.shared = Shared(*args, **kwargs)
		self.encoder = Encoder(shared=self.shared, *args, **kwargs)
		self.decoder = Decoder(shared=self.shared, *args, **kwargs)

	def forward(self, x, batch=None):
		z = self.encoder(x, batch)
		xr, batchr = self.decoder(z)
		return xr, batchr

	def get_vars(self):
		self.vars = {
			"n_pred_logits": self.decoder.get_n_pred_logits(),
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
		size_loss = torch.mean(mean_squared_loss(vars["n_pred_logits"], vars["n"].unsqueeze(-1).detach().float()))
		if torch.isnan(mse_loss):
			mse_loss = 0
		loss = 100 * mse_loss + 1e-3 * size_loss
		corr = correlation(x[tgt_idx], xr[pred_idx])
		return {
			"loss": loss,
			"size_loss": size_loss,
			"mse_loss": mse_loss,
			"corr": corr,
		}



class Encoder(nn.Module):

	def __init__(self, dim, hidden_dim=64, shared=None, **kwargs):
		super().__init__()
		# Params
		self.input_dim = dim
		self.hidden_dim = hidden_dim
		self.shared = shared
		# Modules
		self.val_net = build_mlp(input_dim=self.input_dim, output_dim=self.hidden_dim, nlayers=2, midmult=1.,layernorm=True, nonlinearity=nn.Mish)
		self.rank = torch.nn.Linear(self.input_dim, 1)

	def sort(self, x):
		mag = self.rank(x).reshape(-1)
		_, idx = torch.sort(mag, dim=0)
		return x[idx], idx

	def forward(self, x, batch=None, n_batches=None):
		# x: n x input_dim
		_, input_dim = x.shape
		if batch is None:
			batch = torch.zeros(x.shape[0], device=x.device)
		self.batch = batch

		n = scatter(src=torch.ones(x.shape[0], device=x.device), index=batch, dim_size=n_batches).long()  # batch_size
		ptr = torch.cumsum(torch.cat([torch.zeros(1, device=x.device), n]), dim=0).int()
		self.n = n

		# Zip set start and ends
		xs = []
		xs_idx = []
		for i, j in zip(ptr[:-1], ptr[1:]):
			x_sorted, idx_sorted = self.sort(x[i:j, :])
			xs.append(x_sorted)
			xs_idx.append(idx_sorted + i)
		xs = torch.cat(xs, dim=0) # total_nodes x input_dim
		xs_idx = torch.cat(xs_idx, dim=0)
		self.xs_idx_rev = torch.empty_like(xs_idx, device=x.device).scatter_(0, xs_idx, torch.arange(xs_idx.numel(), device=x.device))
		self.xs = xs
		self.xs_idx = xs_idx

		k = torch.cat([torch.arange(ni, device=x.device) for ni in n], dim=0).int() # batch_size]
		keys = self.shared.get_key(k)

		# Encoder
		y = self.val_net(xs) * keys  # total_nodes x hidden_dim
		z_elements = scatter(src=y, index=batch, dim=-2, dim_size=n_batches)  # batch_size x dim
		n_enc = self.shared.cardinality(n.unsqueeze(-1).float())
		z = z_elements + n_enc
		self.z = z
		return z

	def get_x_perm(self):
		'Returns: the permutation applied to the inputs (shape: ninputs)'
		return self.xs_idx

	def get_z(self):
		'Returns: the latent state (shape: batch x hidden_dim)'
		return self.z

	def get_batch(self):
		'Returns: the batch idxs of the inputs (shape: ninputs)'
		return self.batch

	def get_x(self):
		'Returns: the sorted inputs, x[x_perm] (shape: ninputs x dim)'
		return self.xs

	def get_n(self):
		'Returns: the number of elements per batch (shape: batch)'
		return self.n



class Decoder(nn.Module):

	def __init__(self, dim, hidden_dim=64, shared=None, **kwargs):
		super().__init__()
		# Params
		self.output_dim = dim
		self.hidden_dim = hidden_dim
		self.shared = shared
		# Modules
		self.decoder = build_mlp(input_dim=self.hidden_dim, output_dim=self.output_dim, nlayers=2, midmult=1., layernorm=False, nonlinearity=nn.Mish)
		self.size_pred = build_mlp(input_dim=self.hidden_dim, output_dim=1, nlayers=2, layernorm=True, nonlinearity=nn.Mish)

	def forward(self, z):
		# z: batch_size x hidden_dim
		n_logits = self.size_pred(z.real) # batch_size x 1
		n = torch.round(n_logits, decimals=0).squeeze(-1).int()
		n = torch.maximum(n, torch.tensor(0))
		self.n_pred_logits = n_logits
		self.n_pred = n

		n_enc = self.shared.cardinality(n.unsqueeze(1).float())
		z = z - n_enc

		k = torch.cat([torch.arange(n[i], device=z.device) for i in range(n.shape[0])], dim=0)
		keys = self.shared.get_key(k)
		# breakpoint()
		vals_rep = torch.repeat_interleave(z, n, dim=0)
		zp = vals_rep * keys
		zp_real = zp.real

		x = self.decoder(zp_real)
		batch = torch.repeat_interleave(torch.arange(n.shape[0], device=z.device), n, dim=0)

		self.x = x
		self.batch = batch
		return x, batch

	def get_batch_pred(self):
		'Returns: the batch idxs of the outputs x (shape: noutputs)'
		return self.batch

	def get_x_pred(self):
		'Returns: the outputs x (shape: noutputs x d)'
		return self.x

	def get_n_pred_logits(self):
		'Returns: the class logits for each possible n (shape: batch x 1)'
		return self.n_pred_logits

	def get_n_pred(self):
		'Returns: the actual n, obtained by taking the argmax over n_pred_logits (shape: batch)'
		return self.n_pred



class Shared(nn.Module):

	def __init__(self, dim, hidden_dim, **kwargs):
		super().__init__()
		self.dim = dim
		self.hidden_dim = hidden_dim
		self.cardinality = torch.nn.Linear(1, self.hidden_dim)

	def get_key(self, key):
		t = torch.linspace(0, 1, self.hidden_dim)
		ti = torch.complex(torch.zeros_like(t), t)
		if isinstance(key, int) or isinstance(key, float) or (isinstance(key, torch.Tensor) and key.dim()==0):
			return torch.exp(ti * key * 8)
		elif isinstance(key, torch.Tensor) and key.dim()==1:
			return torch.exp(ti[None,:] * key[:,None] * 8)
		else:
			raise ValueError("key must be 0d or a 1d torch tensor")


if __name__ == '__main__':

	dim = 3
	max_n = 5
	batch_size = 8

	sae = AutoEncoder(dim=dim, hidden_dim=15)
	enc = sae.encoder
	dec = sae.decoder

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

