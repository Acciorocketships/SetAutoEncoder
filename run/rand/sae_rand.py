import torch
from torch.optim import Adam
from torch_geometric.data import Data, Batch
import wandb
import inspect
from sae import AutoEncoderInner, AutoEncoderNew, AutoEncoderVariational
from sae import get_loss_idxs, correlation
from sae.baseline_tspn import AutoEncoder as AutoEncoderTSPN

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="saved/sae_rand-{name}.pt"

project = "sae-rand-test"

def experiments():
	trials = {
		"tspn": {"model": AutoEncoderTSPN, "log": True}
		# "variational-kl": {"model": AutoEncoderVariational, "log": True, "hidden_dim": 64},
		# "new": {"model": AutoEncoderNew},
		# "inner": {"model": AutoEncoderInner},
	}
	default = {
		"dim": 4,
		"hidden_dim": 64,
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
		run(**config)


def run(
			dim = 4,
			hidden_dim = 64,
			max_n = 16,
			epochs = 100000,
			batch_size = 64,
			model_path = model_path_base.format(name="base"),
			model = None,
			name = None,
			load = False,
			log = True,
			**kwargs,
		):

	if inspect.isclass(model):
		model = model(dim=dim, hidden_dim=hidden_dim, max_n=max_n, **kwargs)

	config = kwargs
	config.update({"dim": dim, "hidden_dim": hidden_dim, "max_n": max_n})

	if log:
		wandb.init(
			entity="prorok-lab",
			project=project,
			group=name,
			config=config,
		)

	optim = Adam(model.parameters())

	if load:
		try:
			model_state_dict = torch.load(model_path)
			model.load_state_dict(model_state_dict)
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

		xr, _ = model(data.x, data.batch)

		loss_data = model.loss()
		loss = loss_data["loss"]

		loss.backward()
		optim.step()

		optim.zero_grad()

		var = model.get_vars()
		pred_idx, tgt_idx = get_loss_idxs(var["n_pred"], var["n"])
		corr = correlation(var["x"][tgt_idx], var["xr"][pred_idx])

		if log:
			wandb.log({
				**loss_data,
				"corr": corr,
			})

	if load:
		try:
			model_state_dict = model.state_dict()
			torch.save(model_state_dict, model_path)
		except Exception as e:
			print(e)

	wandb.finish()



if __name__ == '__main__':
	experiments()
