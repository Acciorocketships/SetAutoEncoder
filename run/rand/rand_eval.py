import torch
import wandb
import inspect
from sae.sae_new import AutoEncoder
from sae.baseline_tspn import AutoEncoder as AutoEncoderTSPN
from sae.baseline_dspn import AutoEncoder as AutoEncoderDSPN
from sae.baseline_rnn import AutoEncoder as AutoEncoderRNN
from visualiser import Visualiser

torch.set_printoptions(precision=2, sci_mode=False)
model_path_base="saved/sae_rand-{name}-{hidden_dim}.pt"

project = "sae-rand-eval"

def experiments():
	trials = {
		"sae": [{"model": AutoEncoder}],
		# "rnn": [{"model": AutoEncoderRNN}],
		# "dspn": [{"model": AutoEncoderDSPN}],
		# "tspn": [{"model": AutoEncoderTSPN}],
	}
	default = {
		"hidden_dim": 48,
		"n": 16,
		"log": True,
	}
	for name, trial in trials.items():
		if not isinstance(trial, list):
			trial = [trial]
		for cfg in trial:
			config = default.copy()
			config.update(cfg)
			config["name"] = name
			config["model_path"] = model_path_base.format(name=name, hidden_dim=config["hidden_dim"])
			run(**config)


def run(
			hidden_dim = 96,
			n = 16,
			dim = 6,
			max_n = 16,
			model_path = None,
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
		model_state_dict = torch.load(model_path, map_location=device)
		model.load_state_dict(model_state_dict)
	except Exception as e:
		print(e)

	n = torch.randint(low=1, high=n, size=(1,))[0]
	x = torch.randn(n, dim)
	batch = torch.zeros(n)
	xr, batchr = model(x, batch)

	vis.reset()
	vis.show_objects(xr.cpu().detach())
	vis.show_objects(x.cpu().detach(), alpha=0.5, line={"dash": "dot"})
	plt = vis.render()

	if log:
		wandb.log({"vis": plt})
		wandb.finish()


if __name__ == '__main__':
	experiments()
