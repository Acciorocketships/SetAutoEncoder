import wandb
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from fusion_dataset import ObsEnv
from fusion_model2 import FusionModel
from visualiser2 import Visualiser


project = "fusion"

## Run

def experiments():
	trials = {
		"fusion2": {"log": True},
	}
	default = {
		"feat_dim": 4,
		"pos_dim": 2,
		"batch_size": 64,
		"epochs": 1000000,
		"gnn_nlayers": 3,
		"position": "abs",
		"obs_range": 0.3,
		"mean_objects": 10,
	}
	for name, cfg in trials.items():
		config = default.copy()
		config.update(cfg)
		config["name"] = name
		run(**config)

def run(
			feat_dim = 4,
			pos_dim = 2,
			mean_agents = 8,
			mean_objects = 10,
			epochs = 10000,
			batch_size = 64,
			dataset_size = 1000,
			name = None,
			log = True,
			**kwargs,
		):

	config = locals()

	if log:
		wandb.init(
				entity = "prorok-lab",
				project = project,
				name = name,
				config = config,
			)

	env = ObsEnv(feat_dim=feat_dim, pos_dim=pos_dim, mean_agents=mean_agents, mean_objects=mean_objects, **kwargs)

	vis = Visualiser(visible=False)

	model = FusionModel(input_dim=feat_dim, max_agents=2*mean_agents, max_obj=2*mean_objects, **kwargs)
	optim = Adam(model.parameters())

	i = 0
	for t in range(epochs):

		batch = env.sample_n(dataset_size)
		loader = DataLoader(batch.to_data_list(), batch_size=batch_size, shuffle=True)

		data = None
		for data in loader:

			yhat, batchhat = model(data)
			loss_data = model.loss()

			loss = loss_data["loss"]
			loss.backward()
			optim.step()
			optim.zero_grad()

			if log:
				i += 1
				wandb.log(loss_data, step=i)

		if log:
			imgs = get_layer_rendering(model, data, agent=0, vis=vis)
			wandb.log(imgs, step=i)

	wandb.finish()


def get_layer_rendering(model, data, agent=0, vis=None):
	if vis is None:
		vis = Visualiser(visible=False)
	imgs = {}
	for layer in range(model.gnn_nlayers+1):
		vis.visualise_obs(model, agent=agent, layer=layer, true=True, pred=False)
		true_img = vis.render()
		true_caption = vis.caption
		vis.visualise_obs(model, agent=agent, layer=layer, true=False, pred=True)
		pred_img = vis.render()
		pred_caption = vis.caption
		wandb_trueimg = wandb.Image(true_img, caption=true_caption)
		wandb_predimg = wandb.Image(pred_img, caption=pred_caption)
		imgs["Layer {layer} True".format(layer=layer)] = wandb_trueimg
		imgs["Layer {layer} Pred".format(layer=layer)] = wandb_predimg
	vis.visualise_map(data, agent=agent)
	map_img = vis.render()
	map_caption = vis.caption
	wandb_mapimg = wandb.Image(map_img, caption=map_caption)
	imgs["Full Map"] = wandb_mapimg
	return imgs


def render_model(model, data):
	from torchviz import make_dot
	yhat, batchhat = model(data[0])
	make_dot(yhat, params=dict(list(model.named_parameters()))).render("model", format="png")

if __name__ == '__main__':
	experiments()