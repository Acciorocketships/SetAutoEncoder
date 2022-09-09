import torch
from torch import nn
from sae.util import scatter
from sae.positional import PositionalEncoding
from sae.mlp import build_mlp
from sae.loss import get_loss_idxs, correlation, fixed_order_loss, mean_squared_loss


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
        mse_loss = fixed_order_loss(
            yhat=xr[pred_idx],
            y=x[tgt_idx],
            batch=batch[tgt_idx],
            loss_fn=mean_squared_loss,
        )
        loss = mse_loss
        corr = correlation(x[tgt_idx], xr[pred_idx])
        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "corr": corr,
        }


class Encoder(nn.Module):

    def __init__(self, dim, hidden_dim=64, max_n=8, **kwargs):
        super().__init__()
        # Params
        self.input_dim = dim
        self.weight_dim = int(hidden_dim ** 0.5)
        self.max_n = max_n + 1
        # Modules
        self.size_gen = PositionalEncoding(dim=self.max_n, mode='onehot')
        self.rnn = nn.GRU(input_size=dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)

    def forward(self, x, batch=None):
        # x: n x input_dim
        _, input_dim = x.shape
        if batch is None:
            batch = torch.zeros(x.shape[0])

        n = scatter(src=torch.ones(x.shape[0]), index=batch).long()  # batch_size
        self.n = n
        self.x = x
        self.batch = batch

        max_n = torch.max(n)
        mask = torch.zeros(n.shape[0], max_n).bool()
        ptr = torch.cat([torch.zeros(1), torch.cumsum(n, dim=0)], dim=0).int()
        x_list = [None] * n.shape[0]
        for i in range(n.shape[0]):
            x_list[i] = x[ptr[i]:ptr[i + 1], :]
            mask[i, :n[i]] = True
        x_packed = nn.utils.rnn.pack_sequence(x_list, enforce_sorted=False)

        _, z = self.rnn(x_packed)
        z = z[0, :, :]

        out = torch.cat([z, n.unsqueeze(-1)], dim=-1)
        return out

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
        self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True, num_layers=1)
        self.mapping = build_mlp(input_dim=hidden_dim, output_dim=dim, nlayers=2)

    def forward(self, z):
        # z: batch_size x hidden_dim
        n = z[:, -1].int()
        z = z[:, :-1]
        max_n = torch.max(n)

        y_padded = torch.zeros(z.shape[0], max_n, self.hidden_dim)  # B x N x D
        hidden = z.unsqueeze(0)  # 1 x B x K
        curr = torch.zeros(z.shape[0], 1, self.hidden_dim)  # B x 1 x D
        for i in range(max_n):
            curr, hidden = self.rnn(curr, hidden)
            y_padded[:, i, :] = curr[:, 0, :]

        x_padded = self.mapping(y_padded)

        mask = torch.zeros(n.shape[0], max_n).bool()
        for i in range(n.shape[0]):
            mask[i, :n[i]] = True
        mask_flat = mask.view(n.shape[0] * max_n)
        x_flat_padded = x_padded.view(n.shape[0] * max_n, self.output_dim)
        x = x_flat_padded[mask_flat, :]

        batch = torch.repeat_interleave(torch.arange(n.shape[0]), n, dim=0)
        self.batch = batch
        self.x = x
        self.n = n
        return x, batch

    def get_batch_pred(self):
        'Returns: the batch idxs of the outputs x (shape: noutputs)'
        return self.batch

    def get_x_pred(self):
        'Returns: the outputs x (shape: noutputs x d)'
        return self.x

    def get_n_pred_logits(self):
        'Returns: the class logits for each possible n, up to max_n (shape: batch x max_n)'
        onehot_posgen = PositionalEncoding(dim=self.max_n)
        return onehot_posgen.forward(self.n)

    def get_n_pred(self):
        'Returns: the actual n, obtained by taking the argmax over n_pred_logits (shape: batch)'
        return self.n


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
