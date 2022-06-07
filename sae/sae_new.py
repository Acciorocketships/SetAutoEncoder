import torch
import math
from torch import nn, Tensor
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from sae.mlp import build_mlp


class AutoEncoder(nn.Module):

	def __init__(self, *args, **kwargs):
		super().__init__()
		self.encoder = Encoder(*args, **kwargs)
		self.decoder = Decoder(*args, **kwargs)

	def forward(self, x, batch=None):
		z = self.encoder(x, batch)
		xr, batchr = self.decoder(z)
		return xr, batchr

	def get_vars(self):
		return {
			"n_pred_logits": self.decoder.get_n_pred(),
			"n_pred": self.decoder.get_n(),
			"n": self.encoder.get_n(),
			# input to x permutation
			"x_perm_idx": self.encoder.get_x_perm(),
			"x_unperm_idx": self.encoder.get_x_unperm()
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
		self.key_net_deepset = build_mlp(input_dim=self.max_n, output_dim=self.deepset_dim, nlayers=self.nlayers_keynet, midmult=1., layernorm=True, nonlinearity=self.nonlinearity)
		self.val_net_deepset = build_mlp(input_dim=self.input_dim, output_dim=self.deepset_dim, nlayers=self.nlayers_valnet, midmult=1., layernorm=True, nonlinearity=self.nonlinearity)
		self.encoder_deepset = build_mlp(input_dim=self.deepset_dim, output_dim=self.deepset_dim, nlayers=self.nlayers_encoder, midmult=1., layernorm=self.layernorm, nonlinearity=self.nonlinearity)
		self.key_net_main = build_mlp(input_dim=self.max_n+self.deepset_dim, output_dim=self.hidden_dim, nlayers=self.nlayers_keynet, midmult=1.,layernorm=True, nonlinearity=self.nonlinearity)
		self.val_net_main = build_mlp(input_dim=self.input_dim+self.deepset_dim, output_dim=self.hidden_dim, nlayers=self.nlayers_valnet, midmult=1.,layernorm=True, nonlinearity=self.nonlinearity)
		self.encoder_main = build_mlp(input_dim=self.hidden_dim+self.max_n, output_dim=self.hidden_dim, nlayers=self.nlayers_encoder, midmult=1., layernorm=self.layernorm, nonlinearity=self.nonlinearity)
		self.rank = torch.nn.Linear(self.input_dim, 1)

	def get_x_perm(self):
		return self.xs_idx

	def get_x_unperm(self):
		return self.xs_idx_rev

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

		keys = torch.cat([torch.arange(ni) for ni in n], dim=0).int() # batch_size
		pos = self.pos_gen(keys) # batch_size x hidden_dim

		# Deepset
		y1_ds = self.val_net_deepset(xs) * self.key_net_deepset(pos) # total_nodes x hidden_dim
		y2_ds = scatter(src=y1_ds, index=batch, dim=-2, reduce='sum') # batch_size x dim
		z_ds = self.encoder_deepset(y2_ds) # batch_size x hidden_dim
		z_ds_rep = torch.repeat_interleave(z_ds, n, dim=0)

		# Encoder
		x_in = torch.cat([xs, z_ds_rep], dim=1)
		pos_in = torch.cat([pos, z_ds_rep], dim=1)
		y1 = self.val_net_main(x_in) * self.key_net_main(pos_in)  # total_nodes x hidden_dim
		y2 = scatter(src=y1, index=batch, dim=-2, reduce='sum')  # batch_size x dim
		pos_n = self.pos_gen(n)  # batch_size x max_n
		y3 = torch.cat([y2, pos_n], dim=-1)  # batch_size x (hidden_dim + max_n)
		z = self.encoder_main(y3)  # batch_size x hidden_dim

		return z

	def get_x(self):
		return self.xs

	def get_n(self):
		return self.n


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
		self.nonlinearity = kwargs.get("activation_decoder", nn.ReLU)
		self.pos_mode = kwargs.get("pos_mode", "onehot")
		# Modules
		self.pos_gen = PositionalEncoding(dim=self.max_n, mode=self.pos_mode)
		self.key_net = build_mlp(input_dim=self.max_n, output_dim=self.hidden_dim, nlayers=self.nlayers_keynet, midmult=1., layernorm=True, nonlinearity=self.nonlinearity)
		self.val_net = build_mlp(input_dim=self.hidden_dim, output_dim=self.hidden_dim, nlayers=self.nlayers_valnet, midmult=1., layernorm=True, nonlinearity=self.nonlinearity)
		self.decoder = build_mlp(input_dim=self.hidden_dim, output_dim=self.output_dim, nlayers=self.nlayers_decoder, midmult=1., layernorm=self.layernorm, nonlinearity=self.nonlinearity)
		self.size_pred = build_mlp(input_dim=self.hidden_dim, output_dim=self.max_n, nlayers=2, layernorm=True, nonlinearity=self.nonlinearity)

	def forward(self, z):
		# z: batch_size x hidden_dim
		n_pred = self.size_pred(z) # batch_size x max_n
		self.n_pred = n_pred
		n = torch.argmax(n_pred, dim=-1)
		self.n = n

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
		return x, batch

	def get_n_pred(self):
		return self.n_pred

	def get_n(self):
		return self.n


class PositionalEncoding(nn.Module):

	def __init__(self, dim: int, mode: str = 'onehot'):
		super().__init__()
		self.dim = dim
		self.mode = mode
		self.I = torch.eye(self.dim).byte()
		self.freq_initialised = False

	def forward(self, x: Tensor) -> Tensor:
		if self.mode == 'onehot':
			return self.onehot(x.int()).float()
		elif self.mode == 'binary':
			return self.binary(x.int()).float()
		elif self.mode == 'freq':
			return self.freq(x.int()).float()

	def freq(self, x: Tensor) -> Tensor:
		if not self.freq_initialised:
			self.init_freq()
		out_shape = list(x.shape) + [self.dim]
		return self.pe[x.reshape(-1)].reshape(*out_shape)

	def onehot(self, x: Tensor) -> Tensor:
		out_shape = list(x.shape) + [self.dim]
		return torch.index_select(input=self.I, dim=0, index=x.reshape(-1)).reshape(*out_shape)

	def binary(self, x: Tensor) -> Tensor:
		x = x + 1
		mask = 2 ** torch.arange(self.dim).to(x.device, x.dtype)
		return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

	def binary_to_int(self, x: Tensor) -> Tensor:
		multiplier = 2 ** torch.arange(x.shape[-1]).float().view(-1,1)
		y = x @ multiplier
		return (y-1).int()

def init_freq(self):
		max_len = 2 * self.dim
		position = torch.arange(max_len).unsqueeze(1)
		self.pe = torch.zeros(max_len, self.dim)
		self.pe[:, 0::2] = torch.sin(
			position * torch.exp(torch.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim)))
		self.pe[:, 1::2] = torch.cos(
			position * torch.exp(torch.arange(1, self.dim, 2) * (-math.log(10000.0) / self.dim)))
		self.freq_initialised = True



if __name__ == '__main__':
	from torch.nn.functional import cross_entropy

	dim = 3
	max_n = 5
	batch_size = 16

	# pos = PositionalEncoding(dim=6, mode='binary')
	# k = torch.arange(4)
	# keys = pos(k)
	# n = pos.binary_to_int(keys)
	# y = torch.rand(k.shape[0], 6)
	# loss1 = cross_entropy(y, k, reduction='none')
	# loss2 = cross_entropy(y, keys, reduction='none')

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

