import torch
import wandb
from torch.optim import Adam
from run.fusion.nested.fusion_dataset import ObsEnv
from fusion_model import FusionModel
from visualiser import Visualiser
from sae.util import *

torch.manual_seed(0)

project = "fusion_nested"

## Run

def experiments():
	trials = {
		"filter2tanh": {"log": True},
	}
	default = {
		"feat_dim": 4,
		"pos_dim": 2,
		"batch_size": 64,
		"epochs": 1000000,
		"gnn_nlayers": 3,
		"position": "abs",
		"obs_range": 0.5,
		"mean_objects": 8,
	}
	for name, cfg in trials.items():
		config = default.copy()
		config.update(cfg)
		config["name"] = name
		run(**config)

def run(
			feat_dim = 4,
			pos_dim = 2,
			mean_agents = 3,
			mean_objects = 8,
			max_obj=16,
			epochs = 10000,
			batch_size = 100,
			name = None,
			log = True,
			**kwargs,
		):

	config = locals()

	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda:0"

	if log:
		wandb.init(
				entity = "prorok-lab",
				project = project,
				name = name,
				config = config,
			)

	env = ObsEnv(feat_dim=feat_dim, pos_dim=pos_dim, mean_agents=mean_agents, mean_objects=mean_objects, **kwargs)

	model = FusionModel(input_dim=feat_dim+pos_dim, max_obj=max_obj, **kwargs)
	optim = Adam(model.parameters())

	try:
		model_path = "nested/saved/sae_rand-sae-96.pt"
		model_state_dict = torch.load(model_path, map_location=device)
		model.autoencoder.load_state_dict(model_state_dict)
	except Exception as e:
		print(e)

	if log:
		vis = Visualiser(visible=False)

	i = 0
	for t in range(epochs):

		data = env.sample_n_nested_combined(batch_size)

		yhat, batchhat = model(data)
		loss_data = model.loss()

		loss = loss_data["loss"]
		loss.backward()
		optim.step()
		optim.zero_grad()

		if log:
			wandb.log(loss_data, step=i)
			imgs = get_rendering(model, data, agent=0, vis=vis)
			wandb.log(imgs, step=i)
		i += 1

	wandb.finish()


def get_rendering(model, data, batch=0, agent=0, vis=None):
	if vis is None:
		vis = Visualiser(visible=False)
	imgs = {}
	for layer in range(model.gnn_nlayers):
		vis.visualise_obs(model, data, batch=batch, agent=agent, layer=layer)
		pred_img = vis.render()
		imgs["Layer {layer}".format(layer=layer)] = pred_img
	return imgs

if __name__ == '__main__':
	experiments()