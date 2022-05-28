import torch
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss
from sae import AutoEncoderInner as AutoEncoder
import wandb

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="params/sae_hyper_rand-{name}.pt"
optim_path_base = "params/optim_hyper_rand-{name}.pt"

def experiments():
	trials = {
		# "vanilla": {},
		"no-encode-val": {"encode_val": False},
		"no-encode-val_no-bias": {"encode_val": False, "hypernet_bias": False},
		"layernorm": {"layernorm": True},
		"mag-sort": {"sort": "mag"},
		"hidden-dim-48": {"hidden_dim": 48},
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
			model_path = model_path_base.format(name="base"),
			optim_path = optim_path_base.format(name="base"),
			name = None,
			load = False,
			**kwargs,
		):

	autoencoder = AutoEncoder(dim=dim, hidden_dim=hidden_dim, max_n=max_n, **kwargs)

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

	mse_loss_fn = MSELoss()
	crossentropy_loss_fn = CrossEntropyLoss()

	for t in range(epochs):

		n = torch.randint(low=1, high=max_n, size=(1,))
		x = torch.randn(n[0], dim)

		xr = autoencoder(x)
		var = autoencoder.get_vars()

		x_trunc = var["x"][:min(x.shape[0],xr.shape[0]),:]
		xr_trunc = xr[:min(x.shape[0],xr.shape[0]),:]

		mse_loss = mse_loss_fn(x_trunc, xr_trunc)
		if torch.isnan(mse_loss):
			mse_loss = 0
		crossentropy_loss = crossentropy_loss_fn(var["n_pred"].unsqueeze(0), n)
		loss = mse_loss + crossentropy_loss

		if x_trunc.shape[0]==0:
			corr = 1.
			corr_baseline = 1.
		else:
			corr = np.corrcoef(x_trunc.reshape(-1), xr_trunc.detach().reshape(-1))[0,1]
			corr_baseline = np.corrcoef(x.sum(dim=0).unsqueeze(0).repeat((xr.shape[0], 1)).reshape(-1), xr.detach().reshape(-1))[0,1]


		wandb.log({
					"loss": mse_loss,
					"crossentropy_loss": crossentropy_loss,
					"total_loss": loss,
					"n_correct": n[0]==xr.shape[0],
					"corr": corr,
					"corr_with_sum": corr_baseline,
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