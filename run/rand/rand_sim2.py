import torch
import inspect
from sae.sae_new import AutoEncoder
from sae.baseline_tspn import AutoEncoder as AutoEncoderTSPN
from sae.baseline_dspn import AutoEncoder as AutoEncoderDSPN
from sae.baseline_rnn import AutoEncoder as AutoEncoderRNN
from sae.loss import min_permutation_idxs
from matplotlib import pyplot, cycler

torch.set_printoptions(precision=2, sci_mode=False)
model_name = "sae_rand-{name}-{hidden_dim}"
model_path_base = f"saved/{model_name}.pt"
fig_path_base = "plots/dist.pdf"

seed = 10

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

	dists = []
	names = ["SAE", "RNN", "DSPN", "TSPN"]
	colours = ["#377eb8", "#ff7f00", "#4daf4a", "#f781bf"]
	mindist = None

	for name, trial in trials.items():
		if not isinstance(trial, list):
			trial = [trial]
		for cfg in trial:
			config = default.copy()
			config.update(cfg)
			config["name"] = name
			dists_run, mindists_run = run(**config)
			dists.append(dists_run)
			if name == "sae":
				mindist = mindists_run.mean()

	pyplot_setup()
	fig, ax = pyplot.subplots()
	vplot = ax.violinplot(dists, showmeans=True, showextrema=False)
	for patch, color in zip(vplot['bodies'], colours):
		patch.set_color(color)
	pyplot.axhline(y=mindist, color='r', linestyle='-')
	ax.set_xticks([1,2,3,4])
	ax.set_xticklabels(names)
	pyplot.ylabel("Interpolated Set Arc Length")
	# pyplot.yscale("symlog", linthresh=10, subs=[2,3,4,5,6,7,8,9])
	pyplot.yscale("log")
	pyplot.ylim(bottom=5, top=900)
	pyplot.grid()
	fig.savefig(fig_path_base, bbox_inches="tight")
	pyplot.show()

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

	batchsize = 64
	alpha = torch.linspace(0, 1.0, 100)
	trials = 100
	dists = torch.zeros(trials)
	mindists = torch.zeros(trials)

	torch.manual_seed(seed)

	t = 0
	while t < trials:
		print(name, t)
		data_list = []
		size_list = []
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

		z = model.encoder(x, batch)
		z0 = z[set0_idx,:]
		z1 = z[set1_idx,:]

		z0_interp = z0.unsqueeze(0) * (1-alpha).unsqueeze(1)
		z1_interp = z1.unsqueeze(0) * alpha.unsqueeze(1)
		z_interp = z0_interp + z1_interp

		xr, batchr = model.decoder(z_interp)

		xall = torch.zeros(alpha.shape[0], n, dim)
		try:
			for i in range(alpha.shape[0]):
				xall[i,:,:] = xr[batchr==i,:]
		except:
			continue

		xallr = xall.reshape(alpha.shape[0], n*dim)
		xdiff = xallr[1:,:] - xallr[:-1,:]
		dist = xdiff.norm(dim=1)
		dists[t] = torch.sum(dist)

		x0 = x[batch==set0_idx]
		x1 = x[batch == set1_idx]
		perm = min_permutation_idxs(x1, x0, torch.zeros(x1.shape[0]))
		x1 = x1[perm,:]
		mindist = (x0 - x1).reshape(-1).norm()
		mindists[t] = mindist

		t += 1

	return dists.detach().cpu(), mindists.detach().cpu()


def pyplot_setup():
	tex_fonts = {
		# Use LaTeX to write all text
		"text.usetex": True,
		"font.family": "serif",
		"font.serif": ["Times"],
		# Use 10pt font in plots, to match 10pt font in document
		"axes.labelsize": 16,
		"font.size": 25,
		# Make the legend/label fonts a little smaller
		"legend.fontsize": 12,
		"legend.title_fontsize": 7,
		"legend.framealpha": 0.3,
		"xtick.labelsize": 16,
		"ytick.labelsize": 16,
		# Figure Size
		"figure.figsize": (10, 5),
		# Colour Cycle
		"axes.prop_cycle": cycler(color=["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
										 "#984ea3", "#2bcccc", "#999999", "#e41a1c", "#dede00"])

	}
	pyplot.rcParams.update(tex_fonts)


if __name__ == '__main__':
	experiments()
