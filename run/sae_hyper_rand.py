import torch
from torch.optim import Adam
from sae import AutoEncoderHyper as AutoEncoder
from sae import get_loss_idxs, correlation
from torch.nn import CrossEntropyLoss
import wandb
from torch_geometric.data import Data, Batch

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="params/sae_hyper_rand-{name}.pt"
optim_path_base = "params/optim_hyper_rand-{name}.pt"

def experiments():
	trials = {
		"hyper/vanilla": {},
		# "hyper/no-encode-val": {"encode_val": False},
		# "hyper/non-elementwise": {"elementwise": False},
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

	autoencoder = AutoEncoder(dim=dim,
							  hidden_dim=hidden_dim,
							  max_n=max_n,
							  **kwargs)

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

		xr, _ = autoencoder(data.x, data.batch)
		var = autoencoder.get_vars()

		pred_idx, tgt_idx = get_loss_idxs(
			var["n_pred"], var["n"]
		)

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

	if load:
		try:
			model_state_dict = autoencoder.state_dict()
			torch.save(model_state_dict, model_path)
			optim_state_dict = optim.state_dict()
			torch.save(optim_state_dict, optim_path)
		except Exception as e:
			print(e)

	wandb.finish()



if __name__ == '__main__':
	experiments()