import torch
import numpy as np
from torch.optim import Adam
from torch.nn import MSELoss, CrossEntropyLoss
from sae1 import AutoEncoder
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

torch.set_printoptions(precision=2, sci_mode=False)

dim = 8
max_n = 6
epochs = int(100000)
model_path = "params/sae1.pt"
optim_path = "params/optim1.pt"

autoencoder = AutoEncoder(dim=dim, hidden_dim=64, max_n=max_n)

optim = Adam(autoencoder.parameters())

try:
	model_state_dict = torch.load(model_path)
	autoencoder.load_state_dict(model_state_dict)
	optim_state_dict = torch.load(optim_path)
	optim.load_state_dict(optim_state_dict)
except:
	pass

mse_loss_fn = MSELoss()
crossentropy_loss_fn = CrossEntropyLoss()

tensorboard = SummaryWriter()
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tensorboard.log_dir])
url = tb.launch()

for t in range(epochs):

	n = torch.randint(max_n, size=(1,))
	x = torch.randn(n[0], dim)

	xr = autoencoder(x)
	var = autoencoder.get_vars()

	x_trunc = var["x"][:min(x.shape[0],xr.shape[0]),:]
	xr_trunc = xr[:min(x.shape[0],xr.shape[0]),:]

	mse_loss = mse_loss_fn(x_trunc, xr_trunc)
	if torch.isnan(mse_loss):
		mse_loss = 0
	crossentropy_loss = crossentropy_loss_fn(var["n_pred"].unsqueeze(0), n)
	loss = mse_loss + crossentropy_loss

	if x_trunc.shape[0]==0:
		corr = 1.
		corr_baseline = 1.
	else:
		corr = np.corrcoef(x_trunc.reshape(-1), xr_trunc.detach().reshape(-1))[0,1]
		corr_baseline = np.corrcoef(x.sum(dim=0).unsqueeze(0).repeat((xr.shape[0], 1)).reshape(-1), xr.detach().reshape(-1))[0,1]


	tensorboard.add_scalar('mse_loss', mse_loss, t)
	tensorboard.add_scalar('crossentropy_loss', crossentropy_loss, t)
	tensorboard.add_scalar('loss', loss, t)
	tensorboard.add_scalar('n_correct', n[0]==xr.shape[0], t)
	tensorboard.add_scalar('corr', corr, t)
	tensorboard.add_scalar('corr_baseline', corr_baseline, t)

	loss.backward()
	optim.step()

	optim.zero_grad()

model_state_dict = autoencoder.state_dict()
torch.save(model_state_dict, model_path)
optim_state_dict = optim.state_dict()
torch.save(optim_state_dict, optim_path)

def test(n=4, dim=8):
	x = torch.randn(n,dim)
	xr = autoencoder(x).detach()
	x = autoencoder.get_vars()['x']
	corr = np.corrcoef(x.reshape(-1), xr.reshape(-1))[0,1]
	return {"x": x, "xr": xr, "corr": corr}

test()
# import code; code.interact(local=locals())