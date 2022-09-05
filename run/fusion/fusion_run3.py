import wandb
from torch.optim import Adam
from fusion_dataset import ObsEnv
from fusion_model3 import FusionModel
from visualiser3 import Visualiser


project = "fusion"

## Run

def experiments():
	trials = {
		"fusion3": {"log": False},
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
			batch_size = 100,
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

		data = env.sample_n_nested_combined(batch_size)

		yhat, batchhat = model(data)
		loss_data = model.loss()

		loss = loss_data["loss"]
		loss.backward()
		optim.step()
		optim.zero_grad()

		if log:
			i += 1
			wandb.log(loss_data, step=i)

	wandb.finish()

if __name__ == '__main__':
	experiments()