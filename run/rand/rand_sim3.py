import torch
import wandb
import inspect
from sae.sae_new import AutoEncoder
from sae.baseline_tspn import AutoEncoder as AutoEncoderTSPN
from sae.baseline_dspn import AutoEncoder as AutoEncoderDSPN
from sae.baseline_rnn import AutoEncoder as AutoEncoderRNN
from matplotlib import pyplot

torch.set_printoptions(precision=2, sci_mode=False)
model_name = "sae_rand-{name}-{hidden_dim}"
model_name_n = "sae_rand-{name}-{hidden_dim}-{n}"
model_path_base = f"saved/{model_name}.pt"
fig_path_base = f"plots/{model_name_n}.pdf"

seed = 5

project = "sae-rand-eval"

def experiments():
	trials = {
		"sae": [{"model": AutoEncoder}],
		"rnn": [{"model": AutoEncoderRNN}],
		"dspn": [{"model": AutoEncoderDSPN}],
		"tspn": [{"model": AutoEncoderTSPN}],
	}
	default = {
		"hidden_dim": 96,
		"n": 8,
		"log": False,
	}

	fig, ax = pyplot.subplots()

	for name, trial in trials.items():
		if not isinstance(trial, list):
			trial = [trial]
		for cfg in trial:
			config = default.copy()
			config.update(cfg)
			config["name"] = name
			run(**config)

	ax.legend()
	pyplot.show()


def run(
			hidden_dim = 96,
			dim = 6,
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

	batchsize = 64
	samples = torch.linspace(0, 1.0, 100)
	dists = torch.zeros(samples.shape[0])

	for i, noise in enumerate(samples):
		data_list = []
		size_list = []
		torch.manual_seed(seed)
		for _ in range(batchsize):
			n = torch.randint(low=1, high=max_n, size=(1,))
			x = torch.randn(n[0], dim)
			data_list.append(x)
			size_list.append(n)
		x = torch.cat(data_list, dim=0)
		sizes = torch.cat(size_list, dim=0)
		batch = torch.arange(sizes.numel()).repeat_interleave(sizes)

		eps = torch.randn(x.shape[0], x.shape[1]) * noise
		x_noise = x + eps

		x = x.to(device)
		x_noise = x_noise.to(device)
		batch = batch.to(device)

		z = model.encoder(x, batch)
		z_noise = model.encoder(x_noise, batch)

		dist = (z-z_noise).norm(dim=1).mean()
		dists[i] = dist

	ax = pyplot.gca()
	ax.plot(samples.detach().cpu(), dists.detach().cpu(), label=name)


if __name__ == '__main__':
	experiments()
