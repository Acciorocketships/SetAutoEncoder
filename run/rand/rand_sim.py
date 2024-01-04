import torch
import inspect
from sae.sae_var import AutoEncoder
from sae.baseline_tspn import AutoEncoder as AutoEncoderTSPN
from sae.baseline_dspn import AutoEncoder as AutoEncoderDSPN
from sae.baseline_rnn import AutoEncoder as AutoEncoderRNN
from visualiser import Visualiser

torch.set_printoptions(precision=2, sci_mode=False)
model_name = "sae_rand-{name}"
model_path_base = f"saved/{model_name}.pt"
fig_path_base = "plots/sim/{name}-{alpha}.pdf"

seed = 0

project = "sae-rand-eval"

def experiments():
	trials = {
		"var_dim5_max8_hidden32": [{"model": AutoEncoder}],
		# "rnn": [{"model": AutoEncoderRNN}],
		# "dspn": [{"model": AutoEncoderDSPN}],
		# "tspn": [{"model": AutoEncoderTSPN}],
	}
	default = {
		"hidden_dim": 32,
		"dim": 5,
		"n": 6,
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
			dim = 6,
			n = 8,
			max_n = 16,
			model = None,
			name = None,
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

	try:
		model_path = model_path_base.format(name=name, hidden_dim=config["hidden_dim"])
		model_state_dict = torch.load(model_path, map_location=device)
		model.load_state_dict(model_state_dict)
	except Exception as e:
		print(e)

	vis = Visualiser(visible=False)

	batchsize = 64
	alpha = torch.linspace(0, 1.0, 7)

	data_list = []
	size_list = []
	torch.manual_seed(seed)
	for _ in range(batchsize):
		x = torch.randn(n, dim)
		data_list.append(x)
		size_list.append(n)
	x = torch.cat(data_list, dim=0)
	sizes = torch.tensor(size_list)
	batch = torch.arange(sizes.numel()).repeat_interleave(sizes)

	x = x.to(device)
	batch = batch.to(device)

	set0_idx = 0
	set1_idx = 1

	z, _ = model.encoder(x, batch)
	z0 = z[set0_idx,:]
	z1 = z[set1_idx,:]

	z0_interp = z0.unsqueeze(0) * (1-alpha).unsqueeze(1)
	z1_interp = z1.unsqueeze(0) * alpha.unsqueeze(1)
	z_interp = z0_interp + z1_interp

	xr, batchr = model.decoder(z_interp)

	for i in range(alpha.shape[0]):
		vis.reset()
		if alpha[i] == 0:
			vis.show_objects(x[batch==set0_idx].cpu().detach(), alpha=0.3, linestyle="dashed")
		elif alpha[i] == 1:
			vis.show_objects(x[batch == set1_idx].cpu().detach(), alpha=0.3, linestyle="dashed")
		vis.show_objects(xr[batchr==i].cpu().detach())
		plt = vis.render(lim=2.5)
		vis.save(path=fig_path_base.format(name=name, alpha="%2.2f" % alpha[i]))
		vis.close()



if __name__ == '__main__':
	experiments()
