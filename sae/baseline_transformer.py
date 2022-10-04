import torch
from torch import nn
from sae.util import scatter
from sae.positional import PositionalEncoding
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
        if torch.any(torch.isnan(xr)):
            return {"loss": torch.tensor(float('inf')), "mse_loss":float('inf'), "corr": 0}
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

    def __init__(self, dim, hidden_dim=64, **kwargs):
        super().__init__()
        # Params
        self.input_dim = dim
        self.weight_dim = int(hidden_dim**0.5)
        # Modules
        self.pos_gen = PositionalEncoding(dim=self.input_dim, mode='sinusoid')
        self.Wk = nn.Linear(self.input_dim, self.weight_dim, bias=False)
        self.Wv = nn.Linear(self.input_dim, self.weight_dim, bias=False)

    def forward(self, x, batch=None):
        # x: n x input_dim
        _, input_dim = x.shape
        if batch is None:
            batch = torch.zeros(x.shape[0], device=x.device)

        n = scatter(src=torch.ones(x.shape[0], device=x.device), index=batch).long()  # batch_size
        self.n = n
        self.x = x
        self.batch = batch

        max_n = torch.max(n)
        xmat = torch.zeros(n.shape[0], max_n, self.input_dim, device=x.device)
        mask = torch.zeros(n.shape[0], max_n, device=x.device).bool()
        ptr = torch.cat([torch.zeros(1, device=x.device), torch.cumsum(n, dim=0)], dim=0).int()
        for i in range(n.shape[0]):
            xmat[i, :n[i], :] = x[ptr[i]:ptr[i + 1], :]
            mask[i, :n[i]] = True

        pos = self.pos_gen(torch.arange(max_n, device=x.device)).unsqueeze(0)
        xpos = xmat + pos

        yk = self.Wk(xpos) * mask.unsqueeze(-1)
        yv = self.Wv(xpos) * mask.unsqueeze(-1)
        ykv = yk.unsqueeze(-2) * yv.unsqueeze(-1)
        z_mat = ykv.sum(dim=1)
        z = z_mat.view(n.shape[0], self.weight_dim**2)

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
        self.weight_dim = int(hidden_dim**0.5)
        self.max_n = max_n+1
        # Modules
        self.pos_gen = PositionalEncoding(dim=self.output_dim, mode='sinusoid')
        self.Wq = nn.Linear(self.output_dim, self.weight_dim, bias=False)
        self.mapping = nn.Linear(self.weight_dim, self.output_dim)


    def forward(self, z):
        # z: batch_size x hidden_dim
        n = z[:,-1].int()
        z = z[:,:-1]
        max_n = torch.max(n)

        mask = torch.zeros(n.shape[0], max_n, device=z.device).bool()
        for i in range(n.shape[0]):
            mask[i,:n[i]] = True

        pos = self.pos_gen(torch.arange(max_n, device=z.device))
        query = self.Wq(pos).unsqueeze(0).expand(n.shape[0], -1, -1) # B x N x sqrt(K)
        z_mat = z.view(n.shape[0], self.weight_dim, self.weight_dim)
        decoded = torch.matmul(query, z_mat)
        x_padded = self.mapping(decoded)
        x_flat_padded = x_padded.view(n.shape[0] * max_n, self.output_dim)
        mask_flat = mask.view(n.shape[0] * max_n)
        x = x_flat_padded[mask_flat,:]

        batch = torch.repeat_interleave(torch.arange(n.shape[0], device=z.device), n, dim=0)
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

