import wandb
from torch_geometric.loader import DataLoader
from torch.optim import Adam
from fusion_dataset import ObsEnv
from fusion_model import FusionModel
from visualiser import Visualiser

project = "fusion"

## Run

def experiments():
	trials = {
		"vanilla": {"log": True, "position": "rel"},
	}
	default = {
		"feat_dim": 4,
		"pos_dim": 2,
		"batch_size": 256,
		"epochs": 100000,
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
			loss_type="fixed_order",
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
	vis = Visualiser()

	model = FusionModel(input_dim=feat_dim, max_agents=2*mean_agents, max_obj=2*mean_objects, **kwargs)
	optim = Adam(model.parameters())

	i = 0
	for t in range(epochs):

		batch = env.sample_n(dataset_size)
		loader = DataLoader(batch.to_data_list(), batch_size=batch_size, shuffle=True)

		for data in loader:

			yhat, batchhat = model(data)
			loss_data = model.loss(data)

			loss = loss_data["loss"]
			loss.backward()
			optim.step()
			optim.zero_grad()

			if log:
				i += 1
				wandb.log(loss_data, step=i)

		if log:
			vis.visualise_obs(model, agent=0, layer=0, true=True, pred=False)
			true_img = vis.render()
			true_caption = vis.caption
			vis.visualise_obs(model, agent=0, layer=0, true=False, pred=True)
			pred_img = vis.render()
			pred_caption = vis.caption
			wandb_trueimg = wandb.Image(true_img, caption=true_caption)
			wandb_predimg = wandb.Image(pred_img, caption=pred_caption)
			wandb.log({"Layer 0 True Obj": wandb_trueimg, "Layer 0 Pred Obj": wandb_predimg}, step=i)

	wandb.finish()



if __name__ == '__main__':
	experiments()