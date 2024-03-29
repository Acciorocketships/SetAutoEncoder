import torch
from torch.optim import Adam
import wandb
import inspect
import traceback
from sae import get_loss_idxs, correlation
from sae.sae_ablation import AutoEncoder as AutoEncoderAblation
from visualiser import Visualiser

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="saved/ablation-{name}-{hidden_dim}.pt"

project = "sae-rand-ablation"

def experiments():
	trials = {
		# "sae": [{"model": AutoEncoderAblation, "hidden_dim": 48}, {"model": AutoEncoderAblation, "hidden_dim": 96}],
		# "sae-nocontext": [{"model": AutoEncoderAblation, "hidden_dim": 48, "ablation_context": True}, {"model": AutoEncoderAblation, "hidden_dim": 96, "ablation_context": True}],
		# "sae-nosort": [{"model": AutoEncoderAblation, "hidden_dim": 48, "ablation_sort": True}, {"model": AutoEncoderAblation, "hidden_dim": 96, "ablation_sort": True}],
		# "sae-hungarian": [{"model": AutoEncoderAblation, "hidden_dim": 48, "ablation_hungarian": True}, {"model": AutoEncoderAblation, "hidden_dim": 96, "ablation_hungarian": True}],
		"sae-deepset": [{"model": AutoEncoderAblation, "hidden_dim": 48, "ablation_deepset": True}, {"model": AutoEncoderAblation, "hidden_dim": 96, "ablation_deepset": True}],
	}
	default = {
		"dim": 6,
		"hidden_dim": 48,
		"max_n": 16,
		"epochs": 25000,
		"load": False,
		"save": True,
		"log": True,
		"runs": 10,
		"retries": 1,
	}
	for name, trial in trials.items():
		if not isinstance(trial, list):
			trial = [trial]
		for cfg in trial:
			config = default.copy()
			config.update(cfg)
			config["name"] = name
			config["model_path"] = model_path_base.format(name=name, hidden_dim=config["hidden_dim"])
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
			model_path = "noname.pt",
			model = None,
			name = None,
			load = False,
			save = False,
			log = True,
			**kwargs,
		):

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
		vis = Visualiser(visible=False)

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

			# if t % 100 == 0:
			# 	x0 = x[batch==0]
			# 	xr0 = xr[batchr==0]
			# 	vis.reset()
			# 	vis.show_objects(xr0.cpu().detach())
			# 	vis.show_objects(x0.cpu().detach(), alpha=0.3, linestyle="dashed")
			# 	plt = vis.render()
			# 	log_data["visualisation"] = plt

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
