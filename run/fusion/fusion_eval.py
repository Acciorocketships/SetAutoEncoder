from fusion_dataset import ObsEnv
from fusion_model import FusionModel
from visualiser import Visualiser
from sae.sae_new import AutoEncoder
from sae.util import *

torch.manual_seed(27) # 26, 27

project = "fusion"
model_load_path = "saved/fusion_dim=6_hidden=96_maxn=16_old.pt"

## Run

def experiments():
	trials = {
		"sae": {"autoencoder": AutoEncoder},
	}
	default = {
		"feat_dim": 4,
		"pos_dim": 2,
		"gnn_nlayers": 3,
		"embedding_dim": 96,
		"position": "abs",
		"obs_range": 0.5,
		"com_range": 0.5,
		"max_obj": 12,
		"max_obj_sae": 16,
		"max_agents": 7,
		"log": True,
	}
	for name, cfg in trials.items():
		config = default.copy()
		config.update(cfg)
		run(**config)

def run(
			feat_dim = 4,
			pos_dim = 2,
			obs_range = 0.5,
			com_range = 0.5,
			max_obj = 16,
			max_obj_sae=16,
			max_agents = 6,
			log = True,
			**kwargs,
		):

	config = locals()

	device = "cpu"
	if torch.cuda.is_available():
		device = "cuda:0"

	env = ObsEnv(feat_dim=feat_dim, pos_dim=pos_dim, max_agents=max_agents, max_obj=max_obj, **kwargs)

	model = FusionModel(input_dim=feat_dim+pos_dim, max_obj=max_obj_sae, **kwargs)

	try:
		model_state_dict = torch.load(model_load_path, map_location=device)
		model.load_state_dict(model_state_dict)
	except Exception as e:
		print(e)
		return

	model.eval()

	vis = Visualiser(visible=True)

	data = env.sample_n_nested_combined(1)

	yhat, batchhat = model(data)
	loss_data = model.loss()

	plots = get_rendering(model, data, agent=0, vis=vis)



def get_rendering(model, data, batch=0, agent=0, vis=None):
	if vis is None:
		vis = Visualiser(visible=False)
	imgs = {}
	for layer in range(model.gnn_nlayers):
		vis.visualise_obs(model, data, batch=batch, agent=agent, layer=layer)
		plot = vis.render(lim=[-1, 1, -1, 2], grid=True)
		path = "plots/layer{layer}.pdf".format(layer=layer)
		vis.save(path)
	return imgs


if __name__ == '__main__':
	experiments()