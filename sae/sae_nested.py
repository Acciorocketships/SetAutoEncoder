import torch
from torch import nn
from sae.util import *
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
		self.nonlinearity = kwargs.get("activation_encoder", nn.ReLU)
		self.pos_mode = kwargs.get("pos_mode", "onehot")
		# Modules
		self.pos_gen = PositionalEncoding(dim=self.max_n, mode=self.pos_mode)
		self.key_net_deepset = build_mlp(input_dim=self.max_n, output_dim=self.deepset_dim, nlayers=self.nlayers_keynet, midmult=1., layernorm=False, nonlinearity=self.nonlinearity)
		self.val_net_deepset = build_mlp(input_dim=self.input_dim, output_dim=self.deepset_dim, nlayers=self.nlayers_valnet, midmult=1., layernorm=False, nonlinearity=self.nonlinearity)
		self.encoder_deepset = build_mlp(input_dim=self.deepset_dim, output_dim=self.deepset_dim, nlayers=self.nlayers_encoder, midmult=1., layernorm=False, nonlinearity=self.nonlinearity)
		self.key_net_main = build_mlp(input_dim=self.max_n+self.deepset_dim, output_dim=self.hidden_dim, nlayers=self.nlayers_keynet, midmult=1.,layernorm=False, nonlinearity=self.nonlinearity)
		self.val_net_main = build_mlp(input_dim=self.input_dim+self.deepset_dim, output_dim=self.hidden_dim, nlayers=self.nlayers_valnet, midmult=1.,layernorm=False, nonlinearity=self.nonlinearity)
		self.encoder_main = build_mlp(input_dim=self.hidden_dim+self.max_n, output_dim=self.hidden_dim, nlayers=self.nlayers_encoder, midmult=1., layernorm=False, nonlinearity=self.nonlinearity)
		self.rank = torch.nn.Linear(self.input_dim, 1)

	def sort(self, x):
		mag = self.rank(x)
		perm = torch.nested_tensor([torch.sort(magi, dim=0)[1].squeeze(-1) for magi in mag.unbind()])
		x_sorted = permute_nested(x, perm)
		return x_sorted

	def forward(self, x):
		# x: n x input_dim
		x = self.sort(x)
		n = size_nested(x, dim=1)
		self.x = x
		self.n = n

		# generate keys
		keys = torch.nested_tensor([self.pos_gen(torch.arange(ni)) for ni in n])

		# Deepset
		y1_ds = self.val_net_deepset(x) * self.key_net_deepset(keys)
		y2_ds = sum_nested(y1_ds, dim=-2)
		z_ds = self.encoder_deepset(y2_ds) # batch_size x hidden_dim
		# Encoder
		x_in = cat_nested(x, z_ds.unsqueeze(1), dim=-1)
		pos_in = cat_nested(keys, z_ds.unsqueeze(1), dim=-1)
		y1 = self.val_net_main(x_in) * self.key_net_main(pos_in)  # total_nodes x hidden_dim
		y2 = sum_nested(y1, dim=-2)
		pos_n = self.pos_gen(n)  # batch_size x max_n
		y3 = torch.cat([y2, pos_n], dim=-1)  # batch_size x (hidden_dim + max_n)
		z = self.encoder_main(y3)  # batch_size x hidden_dim
		self.z = z
		return z

	def get_x_perm(self):
		'Returns: the permutation applied to the inputs (shape: ninputs)'
		return self.perm

	def get_z(self):
		'Returns: the latent state (shape: batch x hidden_dim)'
		return self.z

	def get_x(self):
		'Returns: the sorted inputs, x[x_perm] (shape: ninputs x dim)'
		return self.x

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
		self.nlayers_decoder = kwargs.get("nlayers_decoder", 2)
		self.nonlinearity = kwargs.get("activation_decoder", nn.ReLU)
		# Modules
		self.pos_gen = PositionalEncoding(dim=self.max_n, mode="onehot")
		self.key_net = build_mlp(input_dim=self.max_n, output_dim=self.hidden_dim, nlayers=self.nlayers_keynet, midmult=1., layernorm=False, nonlinearity=self.nonlinearity)
		self.decoder = build_mlp(input_dim=self.hidden_dim, output_dim=self.output_dim, nlayers=self.nlayers_decoder, midmult=1., layernorm=False, nonlinearity=self.nonlinearity)
		self.size_pred = build_mlp(input_dim=self.hidden_dim, output_dim=self.max_n, nlayers=2, layernorm=False, nonlinearity=self.nonlinearity)

	def forward(self, z):
		# z: batch_size x hidden_dim
		# size prediction
		n_logits = self.size_pred(z) # batch_size x max_n
		n = self.pos_gen.onehot_logits_to_int(n_logits)
		self.n_pred_logits = n_logits
		self.n_pred = n

		# generate keys
		keys = torch.nested_tensor([self.pos_gen(torch.arange(ni)) for ni in n])
		key_out = self.key_net(keys)

		zp = mul_nested(z, key_out)

		x = self.decoder(zp)

		return x

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

	enc = Encoder(dim=dim, max_n=max_n)
	dec = Decoder(dim=dim, max_n=max_n)

	n = torch.randint(low=0, high=max_n+1, size=(batch_size,))
	total_n = torch.sum(n)
	x_flat = torch.randn(total_n, dim)

	x = torch.nested_tensor(torch.split(x_flat, n.tolist()))

	z = enc(x)
	xr = dec(z)
	print(xr)
