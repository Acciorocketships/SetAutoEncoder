import wandb
from torch.optim import Adam
from fusion_dataset import ObsEnv
from fusion_model import FusionModel
from visualiser import Visualiser
from sae.sae_new import AutoEncoder
from sae.util import *

torch.manual_seed(0)

project = "fusion"

sae_load_path = "saved/sae_rand-sae-96.pt"
model_save_path = "saved/fusion_dim=6_hidden=96_maxn=16.pt"

## Run

def experiments():
	trials = {
		"sae": {"autoencoder": AutoEncoder},
	}
	default = {
		"feat_dim": 4,
		"pos_dim": 2,
		"embedding_dim": 96,
		"batch_size": 64,
		"epochs": 1000,
		"gnn_nlayers": 3,
		"position": "abs",
		"obs_range": 0.5,
		"com_range": 0.5,
		"max_obj": 12,
		"max_obj_sae": 16,
		"max_agents": 5,
		"log": True,
		"load": True,
		"save": True,
	}
	for name, cfg in trials.items():
		config = default.copy()
		config.update(cfg)
		config["name"] = name
		run(**config)

def run(
			feat_dim = 4,
			pos_dim = 2,
			max_agents = 6,
			max_obj = 16,
			max_obj_sae=16,
			epochs = 1000,
			batch_size = 64,
			plot_interval = 10,
			name = None,
			log = True,
			load = True,
			save = True,
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

	env = ObsEnv(feat_dim=feat_dim, pos_dim=pos_dim, max_agents=max_agents, max_obj=max_obj, **kwargs)

	model = FusionModel(input_dim=feat_dim+pos_dim, max_obj=max_obj_sae, **kwargs)
	optim = Adam(model.parameters())

	if load:
		try:
			model_state_dict = torch.load(sae_load_path, map_location=device)
			model.autoencoder.load_state_dict(model_state_dict)
		except Exception as e:
			print(e)

	if log:
		vis = Visualiser(visible=False)

	for t in range(epochs):

		data = env.sample_n_nested_combined(batch_size)

		yhat, batchhat = model(data)
		loss_data = model.loss()

		loss = loss_data["loss"]
		loss.backward()
		optim.step()
		optim.zero_grad()

		if log:
			wandb.log(loss_data, step=t)
			if t % plot_interval == 0:
				imgs = get_rendering(model, data, agent=0, vis=vis)
				wandb.log(imgs, step=t)

	wandb.finish()

	if save:
		try:
			model_state_dict = model.state_dict()
			torch.save(model_state_dict, model_save_path)
		except Exception as e:
			print(e)




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