import torch
from torch.optim import Adam
import wandb
import inspect
import traceback
from sae import get_loss_idxs, correlation
from sae.sae_var import AutoEncoder as VarAutoEncoder
from sae.sae_model import AutoEncoder
from visualiser import Visualiser
from matplotlib import pyplot

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="saved/sae_rand-{name}.pt"

project = "sae-rand-test"

def experiments():
	trials = {
		# "saevar_dim5_max8_hidden32": [{"model": VarAutoEncoder, "dim": 5, "max_n": 8, "hidden_dim": 32, "runs": 1, "log": True, "save": True}],
		"sae_dim5_max8_hidden48": [{"model": AutoEncoder, "dim": 5, "max_n": 8, "hidden_dim": 48, "runs": 1, "log": True, "save": True}],
	}
	default = {
		"dim": 5,
		"hidden_dim": 96,
		"max_n": 16,
		"epochs": 50000,
		"load": False,
		"save": False,
		"log": False,
		"runs": 1,
		"retries": 1,
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
			dim = 5,
			hidden_dim = 64,
			max_n = 16,
			epochs = 50000,
			batch_size = 64,
			model_path = model_path_base.format(name="base"),
			model = None,
			name = None,
			load = False,
			save = False,
			log = True,
			**kwargs,
		):

	data_mean = torch.tensor([0.,0.,0.,0.,-2.]).unsqueeze(0)
	data_var = torch.tensor([2.,3.,1.,1.,2]).unsqueeze(0)

	if inspect.isclass(model):
		model = model(dim=dim, hidden_dim=hidden_dim, max_n=max_n, **kwargs)

	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda:0"

	model = model.to(device=device)

	config = kwargs
	config.update({"dim": dim, "hidden_dim": hidden_dim, "max_n": max_n})

	if log:
		wandb.init(
			entity="prorok-lab",
			project=project,
			group=name,
			config=config,
		)
		vis = Visualiser(visible=True)

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
			x = (torch.rand(n[0], dim) - 0.5) * 2 * data_var + data_mean
			data_list.append(x)
			size_list.append(n)
		x = torch.cat(data_list, dim=0)
		sizes = torch.cat(size_list, dim=0)
		batch = torch.arange(sizes.numel()).repeat_interleave(sizes)

		x = x.to(device)
		batch = batch.to(device)

		xr, batchr = model(x, batch)

		loss_data = model.loss()
		loss = loss_data["loss"]

		loss.backward()
		optim.step()

		optim.zero_grad()

		var = model.get_vars()
		pred_idx, tgt_idx = get_loss_idxs(var["n_pred"], var["n"])
		corr = correlation(var["x"][tgt_idx], var["xr"][pred_idx])

		if log:
			log_data = {
				**loss_data,
				"corr": corr,
			}

			if t % 2500 == 0 and t != 0:
				if isinstance(model, VarAutoEncoder):
					xr, batchr = model(x, batch, sample=False)
				else:
					xr, batchr = model(x, batch)
				x0 = x[batch==0]
				xr0 = xr[batchr==0]
				vis.reset()
				vis.show_objects(xr0.cpu().detach())
				vis.show_objects(x0.cpu().detach(), alpha=0.3, linestyle="--")
				plt = vis.render()
				log_data["visualisation"] = wandb.Image(plt)

			wandb.log(log_data)

	if save:
		try:
			model_state_dict = model.state_dict()
			torch.save(model_state_dict, model_path)
		except Exception as e:
			print(e)

	wandb.finish()



if __name__ == '__main__':
	experiments()
