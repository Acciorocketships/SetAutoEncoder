import torch
from torch import nn
from sae.util import scatter
from sae.mlp import build_mlp
from sae.positional import PositionalEncoding
from sae.loss import get_loss_idxs, correlation
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
		mse_loss = torch.nn.functional.mse_loss(x[tgt_idx], xr[pred_idx])
		crossentropy_loss = CrossEntropyLoss()(vars["n_pred_logits"], vars["n"])
		loss = 100 * mse_loss + crossentropy_loss
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
		self.deepset_dim = dim
		self.hidden_dim = hidden_dim
		self.max_n = max_n + 1
		self.nlayers_keynet = kwargs.get("nlayers_keynet_encoder", 2)
		self.nlayers_valnet = kwargs.get("nlayers_valnet_encoder", 2)
		self.nlayers_encoder = kwargs.get("nlayers_encoder", 2)
		self.layernorm = kwargs.get("layernorm_encoder", False)
		self.nonlinearity = kwargs.get("activation_encoder", nn.Tanh)
		self.pos_mode = kwargs.get("pos_mode", "onehot")
		# Modules
		self.pos_gen = PositionalEncoding(dim=self.max_n, mode=self.pos_mode)
		self.key_net_deepset = build_mlp(input_dim=self.max_n, output_dim=self.deepset_dim, nlayers=self.nlayers_keynet, midmult=1., layernorm=True, nonlinearity=self.nonlinearity)
		self.val_net_deepset = build_mlp(input_dim=self.input_dim, output_dim=self.deepset_dim, nlayers=self.nlayers_valnet, midmult=1., layernorm=True, nonlinearity=self.nonlinearity)
		self.encoder_deepset = build_mlp(input_dim=self.deepset_dim, output_dim=self.deepset_dim, nlayers=self.nlayers_encoder, midmult=1., layernorm=self.layernorm, nonlinearity=self.nonlinearity)
		self.key_net_main = build_mlp(input_dim=self.max_n+self.deepset_dim, output_dim=self.hidden_dim, nlayers=self.nlayers_keynet, midmult=1.,layernorm=True, nonlinearity=self.nonlinearity)
		self.val_net_main = build_mlp(input_dim=self.input_dim+self.deepset_dim, output_dim=self.hidden_dim, nlayers=self.nlayers_valnet, midmult=1.,layernorm=True, nonlinearity=self.nonlinearity)
		self.encoder_main = build_mlp(input_dim=self.hidden_dim+self.max_n, output_dim=self.hidden_dim, nlayers=self.nlayers_encoder, midmult=1., layernorm=self.layernorm, nonlinearity=self.nonlinearity)
		self.rank = torch.nn.Linear(self.input_dim, 1)

	def sort(self, x):
		mag = self.rank(x).reshape(-1)
		_, idx = torch.sort(mag, dim=0)
		return x[idx], idx

	def forward(self, x, batch=None, n_batches=None):
		# x: n x input_dim
		_, input_dim = x.shape
		if batch is None:
			batch = torch.zeros(x.shape[0])
		self.batch = batch

		n = scatter(src=torch.ones(x.shape[0]), index=batch, dim_size=n_batches).long()  # batch_size
		ptr = torch.cumsum(torch.cat([torch.zeros(1), n]), dim=0).int()
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
		self.xs_idx_rev = torch.empty_like(xs_idx).scatter_(0, xs_idx, torch.arange(xs_idx.numel()))
		self.xs = xs
		self.xs_idx = xs_idx

		keys = torch.cat([torch.arange(ni) for ni in n], dim=0).int() # batch_size]
		pos = self.pos_gen(keys) # batch_size x hidden_dim

		# Deepset
		y1_ds = self.val_net_deepset(xs) * self.key_net_deepset(pos) # total_nodes x hidden_dim
		y2_ds = scatter(src=y1_ds, index=batch, dim=-2, dim_size=n_batches) # batch_size x dim
		z_ds = self.encoder_deepset(y2_ds) # batch_size x hidden_dim
		z_ds_rep = torch.repeat_interleave(z_ds, n, dim=0)

		# Encoder
		x_in = torch.cat([xs, z_ds_rep], dim=1)
		pos_in = torch.cat([pos, z_ds_rep], dim=1)
		y1 = self.val_net_main(x_in) * self.key_net_main(pos_in)  # total_nodes x hidden_dim
		y2 = scatter(src=y1, index=batch, dim=-2, dim_size=n_batches)  # batch_size x dim
		pos_n = self.pos_gen(n)  # batch_size x max_n
		y3 = torch.cat([y2, pos_n], dim=-1)  # batch_size x (hidden_dim + max_n)
		z = self.encoder_main(y3)  # batch_size x hidden_dim
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

	def get_max_n(self):
		return self.max_n-1



class Decoder(nn.Module):

	def __init__(self, dim, hidden_dim=64, max_n=8, **kwargs):
		super().__init__()
		# Params
		self.output_dim = dim
		self.hidden_dim = hidden_dim
		self.max_n = max_n + 1
		self.nlayers_keynet = kwargs.get("nlayers_keynet_decoder", 2)
		self.nlayers_valnet = kwargs.get("nlayers_valnet_decoder", 0)
		self.nlayers_decoder = kwargs.get("nlayers_decoder", 2)
		self.layernorm = kwargs.get("layernorm_decoder", False)
		self.nonlinearity = kwargs.get("activation_decoder", nn.Tanh)
		self.pos_mode = kwargs.get("pos_mode", "onehot")
		# Modules
		self.pos_gen = PositionalEncoding(dim=self.max_n, mode=self.pos_mode)
		self.key_net = build_mlp(input_dim=self.max_n, output_dim=self.hidden_dim, nlayers=self.nlayers_keynet, midmult=1., layernorm=True, nonlinearity=self.nonlinearity)
		self.val_net = build_mlp(input_dim=self.hidden_dim, output_dim=self.hidden_dim, nlayers=self.nlayers_valnet, midmult=1., layernorm=True, nonlinearity=self.nonlinearity)
		self.decoder = build_mlp(input_dim=self.hidden_dim, output_dim=self.output_dim, nlayers=self.nlayers_decoder, midmult=1., layernorm=self.layernorm, nonlinearity=self.nonlinearity)
		self.size_pred = build_mlp(input_dim=self.hidden_dim, output_dim=self.max_n, nlayers=2, layernorm=True, nonlinearity=self.nonlinearity)

	def forward(self, z):
		# z: batch_size x hidden_dim
		n_logits = self.size_pred(z) # batch_size x max_n
		if self.pos_mode == "onehot":
			n = self.pos_gen.onehot_logits_to_int(n_logits)
		elif self.pos_mode == "binary":
			b = self.pos_gen.binary_logits_to_binary(n_logits)
			n = self.pos_gen.binary_to_int(b)
		self.n_pred_logits = n_logits
		self.n_pred = n

		k = torch.cat([torch.arange(ni) for ni in n], dim=0)
		pos = self.pos_gen(k) # total_nodes x max_n

		keys = self.key_net(pos)

		if self.nlayers_valnet > 0:
			vals = self.val_net(z)
		else:
			vals = z
		vals_rep = torch.repeat_interleave(vals, n, dim=0)
		zp = vals_rep * keys

		x = self.decoder(zp)
		batch = torch.repeat_interleave(torch.arange(n.shape[0]), n, dim=0)

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
		'Returns: the class logits for each possible n, up to max_n (shape: batch x max_n)'
		return self.n_pred_logits

	def get_n_pred(self):
		'Returns: the actual n, obtained by taking the argmax over n_pred_logits (shape: batch)'
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

