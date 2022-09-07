import torch
from torch.optim import Adam
from sae import EncoderInner, DecoderInner
from sae import AutoEncoderNew as AutoEncoder
from sae import get_loss_idxs, correlation
from torch.nn import CrossEntropyLoss
import wandb
from torch_geometric.data import Data, Batch

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="params/sae_new_rand-{name}.pt"
optim_path_base = "params/optim_new_rand-{name}.pt"

project = "sae-rand"

def experiments():
	trials = {
		"new": {"log": False},
	}
	default = {
		"dim": 4,
		"hidden_dim": 8,
		"max_n": 16,
		"epochs": 100000,
		"load": False,
		"log": True,
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
		encoder = 'new',
		decoder = 'new',
		model_path = model_path_base.format(name="base"),
		optim_path = optim_path_base.format(name="base"),
		name = None,
		load = False,
		log = True,
		**kwargs,
):

	autoencoder = AutoEncoder(dim=dim, hidden_dim=hidden_dim, max_n=max_n, data_batch=True, **kwargs)
	if encoder == 'inner':
		autoencoder.encoder = EncoderInner(dim=dim, hidden_dim=hidden_dim, **kwargs)
	if decoder == 'inner':
		autoencoder.decoder = DecoderInner(hidden_dim=hidden_dim, dim=dim, **kwargs)

	config = kwargs
	config.update({"dim": dim, "hidden_dim": hidden_dim, "max_n": max_n})

	if log:
		wandb.init(
				entity = "prorok-lab",
				project = project,
				group = name,
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

		loss_data = autoencoder.loss()
		loss = loss_data["loss"]

		loss.backward()
		optim.step()

		optim.zero_grad()

		var = autoencoder.get_vars()
		pred_idx, tgt_idx = get_loss_idxs(var["n_pred"], var["n"])
		corr = correlation(x[tgt_idx], xr[pred_idx])

		if log:
			wandb.log({
						**loss_data,
						"corr": corr,
					})


	wandb.finish()



if __name__ == '__main__':
	experiments()
