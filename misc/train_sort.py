import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program

from sort import Rank, AscSort, DescSort

dim = 8
min_n = 6
max_n = 12
epochs = int(100000)

learned_sort = Rank(dim=dim)
truth_sort = AscSort()

optim = Adam(learned_sort.parameters())

mse_loss_fn = MSELoss()

tensorboard = SummaryWriter()
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tensorboard.log_dir])
url = tb.launch()

for t in range(epochs):

	n = torch.randint(low=min_n, high=max_n, size=(1,))
	x = torch.randn(n[0], dim)

	xs = truth_sort(x)
	xs_hat = learned_sort(x)

	loss = mse_loss_fn(xs, xs_hat)

	loss.backward()
	optim.step()

	optim.zero_grad()

	accuracy = torch.sum(truth_sort.indices == learned_sort.indices) / learned_sort.indices.numel()

	tensorboard.add_scalar('loss', loss, t)
	tensorboard.add_scalar('accuracy', accuracy, t)


