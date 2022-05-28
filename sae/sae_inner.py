import torch
import math
from torch import nn, Tensor
from mlp import build_mlp
from matplotlib import pyplot as plt


class AutoEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.encoder = Encoder(*args, **kwargs)
        self.decoder = Decoder(*args, **kwargs)

    def forward(self, x):
        z = self.encoder(x)
        xr = self.decoder(z)
        return xr

    def get_vars(self):
        return {"n_pred": self.decoder.get_n_pred(), "x": self.encoder.get_x()}


class Encoder(nn.Module):

    def __init__(self, dim, hidden_dim=64, max_n=8, **kwargs):
        super().__init__()
        # Params
        self.input_dim = dim
        self.hidden_dim = hidden_dim
        self.max_n = max_n
        # Modules
        self.pos_gen = PositionalEncoding(dim=self.max_n, mode='onehot')
        self.pos_encoder = build_mlp(input_dim=self.max_n, output_dim=self.hidden_dim, nlayers=2, midmult=1., layernorm=False)
        self.enc_psi = build_mlp(input_dim=self.input_dim, output_dim=self.hidden_dim, nlayers=2, midmult=1., layernorm=False)
        self.enc_phi = build_mlp(input_dim=self.hidden_dim+self.max_n, output_dim=self.hidden_dim, nlayers=2, midmult=1., layernorm=False)
        self.rank = torch.nn.Linear(self.input_dim, 1)

    def sort(self, x):
        mag = self.rank(x).reshape(-1)
        _, idx = torch.sort(mag, dim=0)
        return x[idx]

    def forward(self, x):
        # x: n x input_dim
        n, input_dim = x.shape
        xs = self.sort(x)
        pos = self.pos_gen(torch.arange(n)) # n x pos_encoding_dim
        y1 = self.enc_psi(xs) * self.pos_encoder(pos)
        y2 = torch.sum(y1, dim=-2)
        pos_n = self.pos_gen(torch.tensor(n))
        y3 = torch.cat([y2, pos_n], dim=-1)
        z = self.enc_phi(y3)
        self.xs = xs
        return z

    def get_x(self):
        return self.xs

    def get_n(self):
        return self.xs.shape[0]


class Decoder(nn.Module):

    def __init__(self, dim, hidden_dim=64, max_n=8, **kwargs):
        super().__init__()
        # Params
        self.output_dim = dim
        self.hidden_dim = hidden_dim
        self.max_n = max_n
        # Modules
        self.pos_gen = PositionalEncoding(dim=self.max_n, mode='onehot')
        self.pos_encoder = build_mlp(input_dim=self.max_n, output_dim=self.hidden_dim, nlayers=2, midmult=1., layernorm=False)
        self.decoder = build_mlp(input_dim=self.hidden_dim, output_dim=self.output_dim, nlayers=2, midmult=1., layernorm=False)
        self.size_pred = build_mlp(input_dim=self.hidden_dim, output_dim=self.max_n)


    def forward(self, z):
        # z: hidden_dim
        n_pred = self.size_pred(z)
        n = torch.argmax(n_pred)
        pos = self.pos_gen(torch.arange(n)) # n x max_n
        zp = z.unsqueeze(0).expand((n,-1)) * self.pos_encoder(pos) # mult instead?
        x = self.decoder(zp)
        self.n_pred = n_pred
        return x


    def get_n_pred(self):
        return self.n_pred


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
    n = 5
    enc = Encoder(dim=dim)
    dec = Decoder(dim=dim)
    x = torch.rand(n,dim)
    z = enc(x)
    y = dec(z)
    import pdb; pdb.set_trace()

