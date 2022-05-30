import torch
import numpy as np
from torch.optim import Adam
from sae import AutoEncoderInner as AutoEncoder
from sae import mse_sparse, get_loss_idxs, corr_sparse
from torch.nn import CrossEntropyLoss
import wandb
from torch_geometric.data import Data, Batch

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="params/sae_inner_rand-{name}.pt"
optim_path_base = "params/optim_inner_rand-{name}.pt"

def experiments():
	trials = {
		"inner/vanilla": {},
		# "inner/layernorm": {"layernorm": True},
		# "inner/hidden-dim-48": {"hidden_dim": 48},
	}
	default = {
		"dim": 8,
		"hidden_dim": 64,
		"max_n": 6,
		"epochs": 500000,
		"load": False,
	}
	for name, cfg in trials.items():
		config = default.copy()
		config.update(cfg)
		config["name"] = name
		config["model_path"] = model_path_base.format(name=name)
		config["optim_path"] = optim_path_base.format(name=name)
		run(**config)


def run(
			dim = 8,
			hidden_dim = 64,
			max_n = 6,
			epochs = 100000,
			batch_size = 16,
			model_path = model_path_base.format(name="base"),
			optim_path = optim_path_base.format(name="base"),
			name = None,
			load = False,
			**kwargs,
		):

	autoencoder = AutoEncoder(dim=dim, hidden_dim=hidden_dim, max_n=max_n, data_batch=True, **kwargs)

	config = kwargs
	config.update({"dim": dim, "hidden_dim": hidden_dim, "max_n": max_n})

	wandb.init(
			entity = "prorok-lab",
			project = "sae",
			name = name,
			config = config,
		)

	optim = Adam(autoencoder.parameters())

	if load:
		try:
			model_state_dict = torch.load(model_path)
			autoencoder.load_state_dict(model_state_dict)
			optim_state_dict = torch.load(optim_path)
			optim.load_state_dict(optim_state_dict)
		except Exception as e:
			print(e)

	for t in range(epochs):

		data_list = []
		for i in range(batch_size):
			n = torch.randint(low=1, high=max_n, size=(1,))
			x = torch.randn(n[0], dim)
			d = Data(x=x)
			data_list.append(d)
		data = Batch.from_data_list(data_list)

		xr = autoencoder(data)
		var = autoencoder.get_vars()

		pred_idx, tgt_idx = get_loss_idxs(
			var["n_pred_hard"], var["n"]
		)
		mse_loss = torch.nn.functional.mse_loss(var["x"].x[tgt_idx], xr.x[pred_idx])
		crossentropy_loss = CrossEntropyLoss()(var["n_pred"], var["n"])
		loss = mse_loss + crossentropy_loss

		corr = corr_sparse(var["x"], xr)

		wandb.log({
					"loss": mse_loss,
					"crossentropy_loss": crossentropy_loss,
					"total_loss": loss,
					"corr": corr,
				})

		loss.backward()
		optim.step()

		optim.zero_grad()

	if load:
		try:
			model_state_dict = autoencoder.state_dict()
			torch.save(model_state_dict, model_path)
			optim_state_dict = optim.state_dict()
			torch.save(optim_state_dict, optim_path)
		except Exception as e:
			print(e)

	wandb.finish()



def test(n=4, dim=8):
	x = torch.randn(n,dim)
	xr = autoencoder(x).detach()
	x = autoencoder.get_vars()['x']
	corr = np.corrcoef(x.reshape(-1), xr.reshape(-1))[0,1]
	return {"x": x, "xr": xr, "corr": corr}



if __name__ == '__main__':
	experiments()
