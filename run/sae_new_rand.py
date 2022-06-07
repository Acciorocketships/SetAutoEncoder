import torch
from torch.optim import Adam
from sae import EncoderInner, DecoderInner
from sae import AutoEncoderNew as AutoEncoder
from sae import get_loss_idxs, correlation
from torch.nn import CrossEntropyLoss
import wandb
from torch_geometric.data import Data, Batch

torch.set_printoptions(precision=2, sci_mode=False)
project = "sae-rand"

def experiments():
	trials = {
		"encoder=new": {},
	}
	default = {
		"dim": 8,
		"hidden_dim": 32,
		"max_n": 8,
		"epochs": 50000,
	}
	for name, cfg in trials.items():
		config = default.copy()
		config.update(cfg)
		config["name"] = name
		run(**config)


def run(
			dim = 8,
			hidden_dim = 64,
			max_n = 6,
			epochs = 100000,
			batch_size = 16,
			name = None,
			encoder = 'new',
			decoder = 'new',
			**kwargs,
		):

	autoencoder = AutoEncoder(dim=dim, hidden_dim=hidden_dim, max_n=max_n, data_batch=True, **kwargs)
	if encoder == 'inner':
		autoencoder.encoder = EncoderInner(dim=dim, hidden_dim=hidden_dim, **kwargs)
	if decoder == 'inner':
		autoencoder.decoder = DecoderInner(hidden_dim=hidden_dim, dim=dim, **kwargs)

	config = kwargs
	config.update({"dim": dim, "hidden_dim": hidden_dim, "max_n": max_n})

	wandb.init(
			entity = "prorok-lab",
			project = project,
			name = name,
			config = config,
		)

	optim = Adam(autoencoder.parameters())

	for t in range(epochs):

		data_list = []
		for i in range(batch_size):
			n = torch.randint(low=1, high=max_n, size=(1,))
			x = torch.randn(n[0], dim)
			d = Data(x=x)
			data_list.append(d)
		data = Batch.from_data_list(data_list)

		xr, _ = autoencoder(data.x, data.batch)
		var = autoencoder.get_vars()

		pred_idx, tgt_idx = get_loss_idxs(var["n_pred"], var["n"])

		x = data.x[var["x_perm_idx"]]
		mse_loss = torch.nn.functional.mse_loss(x[tgt_idx], xr[pred_idx])
		crossentropy_loss = CrossEntropyLoss()(var["n_pred_logits"], var["n"])
		loss = mse_loss + crossentropy_loss

		corr = correlation(x[tgt_idx], xr[pred_idx])

		wandb.log({
					"loss": mse_loss,
					"crossentropy_loss": crossentropy_loss,
					"total_loss": loss,
					"corr": corr,
				})

		loss.backward()
		optim.step()

		optim.zero_grad()

	wandb.finish()



if __name__ == '__main__':
	experiments()
