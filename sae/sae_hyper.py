import torch
import math
from torch import nn, Tensor
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from sae.mlp import build_mlp, Elementwise
from sae.hyper import update_module_params, get_num_params, replace_hyper_elementwise


class AutoEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, **kwargs)
        self.data_batch = kwargs.get("data_batch", True)

    def forward(self, x, batch=None):
        if self.data_batch:
            data = x
            x = data.x
            batch = data.batch
        z = self.encoder(x, batch)
        xr, batchr = self.decoder(z)
        if self.data_batch:
            return self.create_data_batch(xr, self.decoder.get_n())
        else:
            return xr, batchr

    def get_vars(self):
        if self.data_batch:
            return {
                "n_pred": self.decoder.get_n_pred(),
                "n": self.encoder.get_n(),
                "x": self.create_data_batch(self.encoder.get_x(), self.encoder.get_n())
            }
        else:
            return {
                "n_pred": self.decoder.get_n_pred(),
                "n": self.encoder.get_n(),
                "x": self.encoder.get_x()
            }

    def create_data_batch(self, x, n):
        ptr = torch.cumsum(torch.cat([torch.zeros(1), n]), dim=0).int()
        data_list = [Data(x=x[start_idx:end_idx,:]) for start_idx, end_idx in zip(ptr[:-1], ptr[1:])]
        return Batch.from_data_list(data_list)
        # return Batch(x=x, batch=batch, ptr=ptr)


class Encoder(nn.Module):

	def __init__(self, dim, hidden_dim=64, max_n=8, **kwargs):
		super().__init__()
		# Params
		self.input_dim = dim
		self.hidden_dim = hidden_dim
		self.max_n = max_n
		self.kwargs = kwargs
		# Modules
		self.pos_gen = PositionalEncoding(dim=self.max_n, mode='onehot')
		self.hypernet = Hypernet(key_dim=self.max_n, val_dim=self.hidden_dim, encode_val=kwargs.get("encode_val", True))
		self.enc_psi = build_mlp(input_dim=self.input_dim, output_dim=self.hidden_dim, nlayers=2, midmult=1., layernorm=kwargs.get("layernorm", False))
		self.enc_phi = build_mlp(input_dim=self.hidden_dim+self.max_n, output_dim=self.hidden_dim, nlayers=2, midmult=1., layernorm=kwargs.get("layernorm", False))
		self.rank = torch.nn.Linear(self.input_dim, 1)

	def sort(self, x):
		mag = self.rank(x).reshape(-1)
		_, idx = torch.sort(mag, dim=0)
		return x[idx]

	def forward(self, x, batch=None):
		# x: n x input_dim
		_, input_dim = x.shape
		if batch is None:
			batch = torch.zeros(x.shape[0])

		n = scatter(src=torch.ones(x.shape[0]), index=batch, reduce='sum').long()  # batch_size
		ptr = torch.cumsum(torch.cat([torch.zeros(1), n]), dim=0).int()
		self.n = n

		xs = torch.cat([self.sort(x[i:j, :]) for i, j in zip(ptr[:-1], ptr[1:])], dim=0)  # total_nodes x input_dim
		self.xs = xs

		keys = torch.cat([torch.arange(ni) for ni in n], dim=0).int()  # batch_size
		pos = self.pos_gen(keys)  # batch_size x hidden_dim

		y0 = self.enc_psi(xs)
		self.hypernet.update(key=pos, val=y0)
		y1 = self.hypernet(y0)

		y2 = scatter(src=y1, index=batch, dim=-2, reduce='sum')

		pos_n = self.pos_gen(n)
		y3 = torch.cat([y2, pos_n], dim=-1)

		z = self.enc_phi(y3)
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
		self.max_n = max_n
		# Modules
		self.pos_gen = PositionalEncoding(dim=self.max_n, mode='onehot')
		self.hypernet = Hypernet(key_dim=self.max_n, val_dim=self.hidden_dim, encode_val=False)
		self.decoder = build_mlp(input_dim=self.hidden_dim, output_dim=self.output_dim, nlayers=2, midmult=1., layernorm=kwargs.get("layernorm", False))
		self.size_pred = build_mlp(input_dim=self.hidden_dim, output_dim=self.max_n)

	def forward(self, z):
		# z: batch_size x hidden_dim
		n_pred = self.size_pred(z)  # batch_size x max_n
		self.n_pred = n_pred
		n = torch.argmax(n_pred, dim=-1)
		self.n = n

		keys = torch.cat([torch.arange(ni) for ni in n], dim=0)
		pos = self.pos_gen(keys)  # total_nodes x max_n
		self.hypernet.update(pos)

		z_expanded = torch.repeat_interleave(z, n, dim=0)
		zp = self.hypernet(z_expanded)

		x = self.decoder(zp)

		batch = torch.repeat_interleave(torch.arange(n.shape[0]), n, dim=0)
		return x, batch


	def get_n_pred(self):
		return self.n_pred

	def get_n(self):
		return self.n


class Hypernet(nn.Module):

	def __init__(self, key_dim, val_dim, encode_val=False, **kwargs):
		super().__init__()
		self.encode_val = encode_val
		self.key_dim = key_dim
		self.val_dim = val_dim
		self.key_hidden_dim = self.key_dim
		self.val_hidden_dim = self.val_dim if self.encode_val else 0
		self.hypernet = nn.Sequential(Elementwise(dim=self.val_dim, bias=kwargs.get("hypernet_bias", True)))
		self.num_params = get_num_params(self.hypernet)
		self.paramnet = build_mlp(input_dim=self.key_hidden_dim+self.val_hidden_dim, output_dim=self.num_params, nlayers=2, midmult=1., batchnorm=False)
		self.subnet_key = build_mlp(input_dim=self.key_dim, output_dim=self.key_hidden_dim, nlayers=2, midmult=1., batchnorm=False)
		if self.encode_val:
			self.subnet_val = build_mlp(input_dim=self.val_dim, output_dim=self.val_hidden_dim, nlayers=2, midmult=1., batchnorm=False)


	def update(self, key, val=None):
		key_encoded = self.subnet_key(key)
		if self.encode_val:
			val_encoded = self.subnet_val(val)
			keyval_encoded = torch.cat([key_encoded, val_encoded], dim=-1)
		else:
			keyval_encoded = key_encoded
		params = self.paramnet(keyval_encoded)
		update_module_params(module=self.hypernet, params=params, filter_cond=lambda module: isinstance(module, Elementwise), replace_func=replace_hyper_elementwise)


	def forward(self, x):
		return self.hypernet(x)



class PositionalEncoding(nn.Module):

	def __init__(self, dim: int, mode: str = 'onehot'):
		super().__init__()
		self.dim = dim
		self.mode = mode
		max_len = 2 * dim
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim))
		self.pe = torch.zeros(max_len, self.dim)
		self.pe[:, 0::2] = torch.sin(position * div_term)
		self.pe[:, 1::2] = torch.cos(position * div_term)
		self.I = torch.eye(dim)

	def forward(self, x: Tensor) -> Tensor:
		if self.mode == 'onehot':
			return self.onehot(x)
		elif self.mode == 'freq':
			return self.freq(x)

	def freq(self, x: Tensor) -> Tensor:
		out_shape = list(x.shape) + [self.dim]
		return self.pe[x.reshape(-1)].reshape(*out_shape)

	def onehot(self, x: Tensor) -> Tensor:
		out_shape = list(x.shape) + [self.dim]
		return torch.index_select(input=self.I, dim=0, index=x.reshape(-1)).reshape(*out_shape)




if __name__ == '__main__':

    dim = 3
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

