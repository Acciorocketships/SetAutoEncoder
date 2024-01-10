import torch
import wandb
import inspect
from sae.sae_model import AutoEncoder
from sae.sae_var import AutoEncoder as VarAutoEncoder
from sae.baseline_tspn import AutoEncoder as AutoEncoderTSPN
from sae.baseline_dspn import AutoEncoder as AutoEncoderDSPN
from sae.baseline_rnn import AutoEncoder as AutoEncoderRNN
from visualiser import Visualiser
from sae import get_loss_idxs, correlation

torch.set_printoptions(precision=2, sci_mode=False)
model_name = "sae_rand-{name}"
model_name_n = "sae_rand-{name}"
model_path_base = f"saved/{model_name}.pt"
fig_path_base = f"plots/{model_name_n}.pdf"

seed = 1

project = "sae-rand-eval"

def experiments():
	trials = {
		"sae_dim5_max8_hidden48": [{"model": AutoEncoder, "hidden_dim": 48}],
		"sae_dim5_max8_hidden32": [{"model": AutoEncoder, "hidden_dim": 32}],
		"saevar_dim5_max8_hidden32": [{"model": VarAutoEncoder, "hidden_dim": 32}],
		# "rnn": [{"model": AutoEncoderRNN}],
		# "dspn": [{"model": AutoEncoderDSPN}],
		# "tspn": [{"model": AutoEncoderTSPN}],
	}
	default = {
		"dim": 5,
		"hidden_dim": 48,
		"n": 5,
		"max_n": 8,
		"log": False,
	}
	for name, trial in trials.items():
		if not isinstance(trial, list):
			trial = [trial]
		for cfg in trial:
			config = default.copy()
			config.update(cfg)
			config["name"] = name
			run(**config)


def run(
			hidden_dim = 96,
			n = 16,
			dim = 6,
			max_n = 16,
			model = None,
			name = None,
			log = True,
			**kwargs,
		):

	if inspect.isclass(model):
		model = model(dim=dim, hidden_dim=hidden_dim, max_n=max_n, **kwargs)

	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda:0"

	model = model.to(device=device)
	model.eval()

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

	try:
		model_path = model_path_base.format(name=name)
		model_state_dict = torch.load(model_path, map_location=device)
		model.load_state_dict(model_state_dict)
	except Exception as e:
		print(e)

	data_mean = torch.tensor([0.,0.,0.,0.,-2.]).unsqueeze(0)
	data_var = torch.tensor([2.,3.,1.,1.,2]).unsqueeze(0)

	data_list = []
	size_list = []
	torch.manual_seed(seed)
	for i in range(64):
		x = (torch.rand(n, dim) - 0.5) * 2 * data_var + data_mean
		data_list.append(x)
		size_list.append(n)
	x = torch.cat(data_list, dim=0)
	sizes = torch.tensor(size_list)
	batch = torch.arange(sizes.numel()).repeat_interleave(sizes)

	x = x.to(device)
	batch = batch.to(device)

	if isinstance(model, VarAutoEncoder):
		xr, batchr = model(x, batch, sample=False)
	else:
		xr, batchr = model(x, batch)

	x = x[batch==0]
	xr = xr[batchr==0]

	var = model.get_vars()
	pred_idx, tgt_idx = get_loss_idxs(var["n_pred"], var["n"])
	corr = correlation(var["x"][tgt_idx], var["xr"][pred_idx])
	print(f"{name}: corr={corr}")

	vis.reset()
	vis.show_objects(x.cpu().detach(), alpha=0.3, linestyle="dashed")
	vis.show_objects(xr.cpu().detach())
	plt = vis.render()
	vis.save(path=fig_path_base.format(name=name))

	if log:
		wandb.log({"vis": plt})
		wandb.finish()


if __name__ == '__main__':
	experiments()
