import torch
import inspect
from sae.sae_new import AutoEncoder
from sae.baseline_tspn import AutoEncoder as AutoEncoderTSPN
from sae.baseline_dspn import AutoEncoder as AutoEncoderDSPN
from sae.baseline_rnn import AutoEncoder as AutoEncoderRNN
from matplotlib import pyplot, cycler
from sae.loss import correlation, get_loss_idxs

torch.set_printoptions(precision=2, sci_mode=False)
model_name = "{name}-{hidden_dim}"
model_path_base = f"saved/sae_rand-{model_name}.pt"
fig_path_base = "plots/corr_scale-{hidden_dim}.pdf"

seed = 0



def experiments():
	trials = {
		"sae": [{"model": AutoEncoder, "label": "PISA"}],
		"rnn": [{"model": AutoEncoderRNN, "label": "GRU"}],
		"dspn": [{"model": AutoEncoderDSPN, "label": "DSPN"}],
		"tspn": [{"model": AutoEncoderTSPN, "label": "TSPN"}],
	}
	default = {
		"hidden_dim": 96,
		"max_n": 16,
	}

	pyplot_setup()
	fig, ax = pyplot.subplots()

	for name, trial in trials.items():
		if not isinstance(trial, list):
			trial = [trial]
		for cfg in trial:
			config = default.copy()
			config.update(cfg)
			config["name"] = name
			run(**config)

	ax.legend(loc="lower left")
	pyplot.ylim(bottom=0., top=1.02)
	pyplot.xlim(left=0, right=16)
	pyplot.xlabel("Number of Elements")
	pyplot.ylabel("Correlation")
	fig.savefig(fig_path_base.format(hidden_dim=default["hidden_dim"]), bbox_inches="tight")
	pyplot.show()

def run(
			hidden_dim = 96,
			dim = 6,
			max_n = 16,
			model = None,
			name = None,
			label = None,
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

	batchsize = 1024
	ns = torch.arange(0, max_n+1)
	corrs = torch.ones(ns.shape[0])
	corr_mins = torch.ones(ns.shape[0])
	corr_maxs = torch.ones(ns.shape[0])

	for i, n in enumerate(ns[1:]):
		data_list = []
		size_list = []
		torch.manual_seed(seed)
		for _ in range(batchsize):
			x = torch.randn(n, dim)
			data_list.append(x)
			size_list.append(n)
		x = torch.cat(data_list, dim=0)
		sizes = torch.stack(size_list, dim=0)
		batch = torch.arange(sizes.numel()).repeat_interleave(sizes)

		x = x.to(device)
		batch = batch.to(device)

		xr, batchr = model(x, batch)

		var = model.get_vars()
		pred_idx, tgt_idx = get_loss_idxs(var["n_pred"], var["n"])
		corr_arr = correlation(var["x"][tgt_idx], var["xr"][pred_idx], return_arr=True)
		corr = torch.mean(corr_arr)
		corr_min = torch.min(corr_arr)
		corr_max = torch.max(corr_arr)

		corrs[i+1] = corr
		corr_mins[i+1] = corr_min
		corr_maxs[i+1] = corr_max

	ax = pyplot.gca()
	(mean_line,) = ax.plot(ns.detach().cpu(), corrs.detach().cpu(), label=label)
	ax.fill_between(ns.detach().cpu(), corr_mins.detach().cpu(), corr_maxs.detach().cpu(), color=mean_line.get_color(), alpha=0.3)


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
