import torch
from torch import nn
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from sae.mlp import build_mlp
from sae.positional import PositionalEncoding
from sae.loss import get_loss_idxs, correlation, min_permutation_loss, mean_squared_loss
from sae.util import collect
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
        # Modules
        self.pos_gen = PositionalEncoding(dim=self.max_n, mode=kwargs.get('pe', 'onehot'))
        self.rank = torch.nn.Linear(self.input_dim, 1)
        self.enc = build_mlp(input_dim=self.input_dim, output_dim=self.hidden_dim, nlayers=2, midmult=1.,
                             layernorm=True, nonlinearity=nn.Tanh)
        self.transformer = LinearAttentionBlock(self.input_dim, self.hidden_dim)


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

        max_n = torch.max(n)
        y = torch.zeros(n.shape[0], max_n, self.input_dim)
        ptr = torch.cat([torch.zeros(1), torch.cumsum(n, dim=0)], dim=0).int()
        # mask = torch.zeros(n.shape[0], max_n).bool()
        for i in range(n.shape[0]):
            y[i, :n[i], :] = x[ptr[i]:ptr[i + 1], :]
            # mask[i, :n[i]] = True

        z_padded = self.transformer(y)[0]
        # mask_flat = mask.view(n.shape[0] * max_n)
        # z_all = z_padded.view(n.shape[0] * max_n, self.hidden_dim)[mask_flat, :]
        z = collect(input=z_padded, index=n-1, dim_along=0, dim_select=1)
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


    def forward(self, z):
        # z: batch_size x hidden_dim
        n_pred = self.size_pred(z)  # batch_size x max_n
        n = torch.argmax(n_pred, dim=-1)
        self.n_pred_logits = n_pred
        self.n_pred = n


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


class LinearAttentionBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.key = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query = nn.Linear(input_dim, hidden_dim, bias=False)
        self.value = nn.Linear(input_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(input_dim)
        self.phi = Phi()

    def forward(self, x, state=None):
        if state is None:
            state = self.initial_state(x.shape[0])
        x = self.norm(x)
        K = self.phi(self.key(x))
        Q = self.phi(self.query(x))
        V = self.value(x)
        S, Z = state
        B, T, F = K.shape

        # S = sum(K V^T)
        S = S.reshape(B, 1, F, F) + torch.einsum("bti, btj -> btij", K, V).cumsum(dim=1)
        # Z = sum(K)
        Z = Z.reshape(B, 1, F) + K.cumsum(dim=1)
        # numerator = Q^T S
        # numerator = torch.einsum("btoi, btil -> btol", Q.unsqueeze(-2), S).squeeze(-2)
        numerator = torch.einsum("bti, btil -> btl", Q, S)
        # denominator = Q^T Z
        denominator = torch.einsum("bti, btl -> bt", Q, Z).reshape(B, T, 1) + 1e-5
        # output = (Q^T S) / (Q^T Z)
        output = numerator / denominator
        state = [S, Z]
        return output, state

    def initial_state(self, batch=1):
        return [
            torch.zeros(batch, self.hidden_dim, self.hidden_dim),
            torch.zeros(batch, self.hidden_dim),
        ]

class Phi(nn.Module):
    def forward(self, x):
        return torch.nn.functional.elu(x) + 1


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

