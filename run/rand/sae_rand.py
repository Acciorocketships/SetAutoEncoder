import torch
from torch.optim import Adam
import wandb
import inspect
import traceback
from sae import AutoEncoderInner, AutoEncoderNew, AutoEncoderVariational
from sae import get_loss_idxs, correlation
from sae.baseline_tspn import AutoEncoder as AutoEncoderTSPN
from sae.baseline_transformer import AutoEncoder as AutoEncoderTransformer
from sae.baseline_rnn import AutoEncoder as AutoEncoderRNN

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="saved/sae_rand-{name}.pt"

project = "sae-rand"

def experiments():
	trials = {
		"rnn": [{"model": AutoEncoderRNN}, {"model": AutoEncoderRNN, "hidden_dim": 32}],
		"transformer": [{"model": AutoEncoderTransformer}, {"model": AutoEncoderTransformer, "hidden_dim": 32}],
		"tspn": [{"model": AutoEncoderTSPN}, {"model": AutoEncoderTSPN, "hidden_dim": 32}],
		"sae": [{"model": AutoEncoderNew}, {"model": AutoEncoderNew, "hidden_dim": 32}],
		# "variational-kl": {"model": AutoEncoderVariational, "log": True, "hidden_dim": 64},
	}
	default = {
		"dim": 4,
		"hidden_dim": 64,
		"max_n": 16,
		"epochs": 25000,
		"load": False,
		"log": True,
		"runs": 10,
		"retries": 3,
	}
	for name, trial in trials.items():
		if not isinstance(trial, list):
			trial = [trial]
		for cfg in trial:
			config = default.copy()
			config.update(cfg)
			config["name"] = name
			config["model_path"] = model_path_base.format(name=name)
			for run_num in range(config["runs"]):
				for retry in range(1,config["retries"]+1):
					try:
						run(**config)
						break
					except Exception as e:
						print("Trial {retry} failed:".format(retry=retry))
						print(traceback.format_exc())
						if retry < config["retries"]:
							print("Retrying...")


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
		size_list = []
		for i in range(batch_size):
			n = torch.randint(low=1, high=max_n, size=(1,))
			x = torch.randn(n[0], dim)
			data_list.append(x)
			size_list.append(n)
		data = torch.cat(data_list, dim=0)
		sizes = torch.cat(size_list, dim=0)
		batch = torch.arange(sizes.numel()).repeat_interleave(sizes)

		xr, _ = model(data, batch)

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
