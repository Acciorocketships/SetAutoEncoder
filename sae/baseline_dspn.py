import torch
from torch import nn
import scipy
from sae.util import scatter
from sae.loss import get_loss_idxs, correlation, min_permutation_idxs, mean_squared_loss


class AutoEncoder(nn.Module):

	def __init__(self, *args, **kwargs):
		'''
		Must have self.encoder and self.decoder objects, which follow the encoder and decoder interfaces
		'''
		super().__init__()
		self.encoder = Encoder(*args, **kwargs)
		self.decoder = Decoder(self.encoder, *args, **kwargs)

	def forward(self, x, batch=None):
		z = self.encoder(x, batch)
		xr, batchr = self.decoder(z)
		return xr, batchr

	def get_vars(self):
		self.vars = {
			"n_pred": self.decoder.get_n_pred(),
			"n": self.encoder.get_n(),
			"batch": self.encoder.get_batch(),
			"x": self.encoder.get_x(),
			"xr": self.decoder.get_x_pred(),
			"xmat": self.encoder.get_xmat(),
			"xmatr": self.decoder.get_xmat(),
			"mask": self.encoder.get_mask(),
			"maskr": self.decoder.get_mask(),
			"repr_loss": self.decoder.get_repr_loss(),
		}
		return self.vars

	def loss(self, vars=None):
		'''
		Input: the output of self.get_vars()
		Returns: a dict of info which must include 'loss'
		'''
		if vars is None:
			vars = self.get_vars()

		target_set = vars["xmat"]
		target_mask = vars["mask"]
		pred_set = vars["xmatr"]
		pred_mask = vars["maskr"]

		target_set = torch.cat(
			[target_set, target_mask.unsqueeze(dim=-1)], dim=-1
		)
		pred_set = torch.cat(
			[pred_set, pred_mask.unsqueeze(dim=-1)], dim=-1
		)

		set_loss = hungarian_loss(pred_set.permute(0,2,1), target_set.permute(0,2,1)).unsqueeze(0)

		loss = set_loss.mean()

		pred_idx, tgt_idx = get_loss_idxs(vars["n_pred"], vars["n"])
		x = vars["x"][tgt_idx]
		xr = vars["xr"][pred_idx]
		batch = vars["batch"][tgt_idx]
		perm = min_permutation_idxs(
			yhat=xr,
			y=x,
			batch=batch,
			loss_fn=mean_squared_loss,
		)
		vars["perm"] = perm
		xr = xr[perm]
		mse_loss = torch.mean(mean_squared_loss(x, xr))
		corr = correlation(x, xr)
		return {
			"loss": loss,
			"mse_loss": mse_loss,
			"corr": corr,
		}



class Encoder(nn.Module):

	def __init__(self, dim, hidden_dim, max_n, **kwargs):
		super().__init__()
		# Params
		self.input_dim = dim
		self.hidden_dim = hidden_dim
		self.max_n = max_n
		self.encoder = FSEncoder(input_channels=dim, output_channels=hidden_dim)

	def forward(self, x, batch=None):
		# x: n x input_dim
		# x: n x input_dim
		_, input_dim = x.shape
		if batch is None:
			batch = torch.zeros(x.shape[0], device=x.device)

		n = scatter(src=torch.ones(x.shape[0], device=x.device), index=batch).long()  # batch_size
		self.n = n
		self.x = x
		self.batch = batch

		xmat = torch.zeros(n.shape[0], self.max_n, self.input_dim, device=x.device)
		mask = torch.zeros(n.shape[0], self.max_n, device=x.device).bool()
		ptr = torch.cat([torch.zeros(1, device=x.device), torch.cumsum(n, dim=0)], dim=0).int()
		for i in range(n.shape[0]):
			xmat[i, :n[i], :] = x[ptr[i]:ptr[i + 1], :]
			mask[i, :n[i]] = True
		self.mask = mask
		self.xmat = xmat

		xmat = xmat.permute(0,2,1)
		z = self.encoder(xmat, mask)
		self.z = z
		return z

	def get_x_perm(self):
		'Returns: the permutation applied to the inputs (shape: ninputs)'
		return torch.arange(self.x.shape[0])

	def get_z(self):
		'Returns: the latent state (shape: batch x hidden_dim)'
		return self.z

	def get_mask(self):
		return self.mask

	def get_xmat(self):
		return self.xmat

	def get_batch(self):
		'Returns: the batch idxs of the inputs (shape: ninputs)'
		return self.batch

	def get_x(self):
		'Returns: the sorted inputs, x[x_perm] (shape: ninputs x d)'
		return self.x

	def get_n(self):
		'Returns: the number of elements per batch (shape: batch)'
		return self.n


class Decoder(nn.Module):

	def __init__(self, encoder, dim, max_n, **kwargs):
		super().__init__()
		# Params
		self.decoder = DSPN(encoder=encoder.encoder, set_channels=dim, max_set_size=max_n)

	def forward(self, z):
		# z: batch_size x hidden_dim
		xr, mask = self.decoder(z)
		xr = xr.permute(0,2,1)
		mask_binary = mask > 0.5
		batch = torch.arange(xr.shape[0], device=z.device).unsqueeze(1).expand(-1, xr.shape[1])
		xr_flat = xr.reshape(xr.shape[0] * xr.shape[1], xr.shape[2])
		batch_flat = batch.reshape(batch.shape[0] * batch.shape[1])
		mask_flat = mask_binary.view(mask.shape[0] * mask.shape[1])
		xr_out = xr_flat[mask_flat,:]
		batch_out = batch_flat[mask_flat]
		self.x = xr_out
		self.batch = batch_out
		self.n_pred = mask_binary.sum(dim=1)
		return xr_out, batch_out

	def get_batch_pred(self):
		'Returns: the batch idxs of the outputs x (shape: noutputs)'
		return self.batch

	def get_x_pred(self):
		'Returns: the outputs x (shape: noutputs x d)'
		return self.x

	def get_n_pred(self):
		'Returns: the actual n, obtained by taking the argmax over n_pred_logits (shape: batch)'
		return self.n_pred

	def get_repr_loss(self):
		return self.decoder.get_loss()

	def get_xmat(self):
		return self.decoder.get_pred_iter()[-1].permute(0,2,1)

	def get_mask(self):
		return self.decoder.get_mask_iter()[-1]


class DSPN(nn.Module):
	""" Deep Set Prediction Networks
	Yan Zhang, Jonathon Hare, Adam Prügel-Bennett
	https://arxiv.org/abs/1906.06565
	"""

	def __init__(self, encoder, set_channels, max_set_size, iters=30, lr=800):
		"""
		encoder: Set encoder module that takes a set as input and returns a representation thereof.
			It should have a forward function that takes two arguments:
			- a set: FloatTensor of size (batch_size, input_channels, maximum_set_size). Each set
			should be padded to the same maximum size with 0s, even across batches.
			- a mask: FloatTensor of size (batch_size, maximum_set_size). This should take the value 1
			if the corresponding element is present and 0 if not.
		channels: Number of channels of the set to predict.
		max_set_size: Maximum size of the set.
		iter: Number of iterations to run the DSPN algorithm for.
		lr: Learning rate of inner gradient descent in DSPN.
		"""
		super().__init__()
		self.encoder = encoder
		self.iters = iters
		self.lr = lr

		self.starting_set = nn.Parameter(torch.rand(1, set_channels, max_set_size))
		self.starting_mask = nn.Parameter(0.5 * torch.ones(1, max_set_size))

	def forward(self, target_repr):
		"""
		Conceptually, DSPN simply turns the target_repr feature vector into a set.
		target_repr: Representation that the predicted set should match. FloatTensor of size (batch_size, repr_channels).
		This can come from a set processed with the same encoder as self.encoder (auto-encoder), or a different
		input completely (normal supervised learning), such as an image encoded into a feature vector.
		"""
		# copy same initial set over batch
		current_set = self.starting_set.expand(
			target_repr.size(0), *self.starting_set.size()[1:]
		)
		current_mask = self.starting_mask.expand(
			target_repr.size(0), self.starting_mask.size()[1]
		)
		# make sure mask is valid
		current_mask = current_mask.clamp(min=0, max=1)

		# info used for loss computation
		intermediate_sets = [current_set]
		intermediate_masks = [current_mask]
		# info used for debugging
		repr_losses = []
		grad_norms = []

		# optimise repr_loss for fixed number of steps
		for i in range(self.iters):
			# regardless of grad setting in train or eval, each iteration requires torch.autograd.grad to be used
			with torch.enable_grad():
				if not self.training:
					current_set.requires_grad_(True)
					current_mask.requires_grad_(True)

				# compute representation of current set
				predicted_repr = self.encoder(current_set, current_mask)
				# how well does the representation matches the target
				repr_loss = torch.nn.functional.smooth_l1_loss(
					predicted_repr, target_repr, reduction="mean"
				)
				# change to make to set and masks to improve the representation
				set_grad, mask_grad = torch.autograd.grad(
					inputs=[current_set, current_mask],
					outputs=repr_loss,
					only_inputs=True,
					create_graph=True,
				)
			# update set with gradient descent
			current_set = current_set - self.lr * set_grad
			current_mask = current_mask - self.lr * mask_grad
			current_mask = current_mask.clamp(min=0, max=1)
			# save some memory in eval mode
			if not self.training:
				current_set = current_set.detach()
				current_mask = current_mask.detach()
				repr_loss = repr_loss.detach()
				set_grad = set_grad.detach()
				mask_grad = mask_grad.detach()
			# keep track of intermediates
			intermediate_sets.append(current_set)
			intermediate_masks.append(current_mask)
			repr_losses.append(repr_loss)
			grad_norms.append(set_grad.norm())

		self.repr_losses = repr_losses
		self.pred_iter = intermediate_sets
		self.mask_iter = intermediate_masks
		return intermediate_sets[-1], intermediate_masks[-1]

	def get_loss(self):
		return sum(self.repr_losses)

	def get_pred_iter(self):
		return self.pred_iter

	def get_mask_iter(self):
		return self.mask_iter


class FSEncoder(nn.Module):
	def __init__(self, input_channels, output_channels):
		super().__init__()
		hidden_dim = (input_channels + output_channels) // 2
		self.conv = nn.Sequential(
			nn.Conv1d(input_channels + 1, hidden_dim, 1),
			nn.ReLU(),
			nn.Conv1d(hidden_dim, hidden_dim, 1),
			nn.ReLU(),
			nn.Conv1d(hidden_dim, output_channels, 1),
		)
		self.pool = FSPool(output_channels, 20, relaxed=False)

	def forward(self, x, mask=None):
		mask = mask.unsqueeze(1)
		x = torch.cat([x, mask], dim=1)  # include mask as part of set
		x = self.conv(x)
		x = x / x.size(2)  # normalise so that activations aren't too high with big sets
		x, _ = self.pool(x)
		return x


class FSPool(nn.Module):
	"""
		Featurewise sort pooling. From:
		FSPool: Learning Set Representations with Featurewise Sort Pooling.
	"""
	def __init__(self, in_channels, n_pieces, relaxed=False):
		"""
		in_channels: Number of channels in input
		n_pieces: Number of pieces in piecewise linear
		relaxed: Use sorting networks relaxation instead of traditional sorting
		"""
		super().__init__()
		self.n_pieces = n_pieces
		self.weight = nn.Parameter(torch.zeros(in_channels, n_pieces + 1))
		self.relaxed = relaxed
		self.reset_parameters()

	def reset_parameters(self):
		nn.init.normal_(self.weight)

	def forward(self, x, n=None):
		""" FSPool
		x: FloatTensor of shape (batch_size, in_channels, set size).
		This should contain the features of the elements in the set.
		Variable set sizes should be padded to the maximum set size in the batch with 0s.
		n: LongTensor of shape (batch_size).
		This tensor contains the sizes of each set in the batch.
		If not specified, assumes that every set has the same size of x.size(2).
		Note that n.max() should never be greater than x.size(2), i.e. the specified set size in the
		n tensor must not be greater than the number of elements stored in the x tensor.
		Returns: pooled input x, used permutation matrix perm
		"""
		assert x.size(1) == self.weight.size(0), 'incorrect number of input channels in weight'
		# can call withtout length tensor, uses same length for all sets in the batch
		if n is None:
			n = x.new(x.size(0)).fill_(x.size(2)).long()
		# create tensor of ratios $r$
		sizes, mask = fill_sizes(n, x)
		mask = mask.expand_as(x)
		# turn continuous into concrete weights
		weight = self.determine_weight(sizes)
		# make sure that fill value isn't affecting sort result
		# sort is descending, so put unreasonably low value in places to be masked away
		x = x + (1 - mask).float() * -99999
		if self.relaxed:
			x, perm = cont_sort(x, temp=self.relaxed)
		else:
			x, perm = x.sort(dim=2, descending=True)
		x = (x * weight * mask.float()).sum(dim=2)
		return x, perm

	def forward_transpose(self, x, perm, n=None):
		""" FSUnpool
		x: FloatTensor of shape (batch_size, in_channels)
		perm: Permutation matrix returned by forward function.
		n: LongTensor fo shape (batch_size)
		"""
		if n is None:
			n = x.new(x.size(0)).fill_(perm.size(2)).long()
		sizes, mask = fill_sizes(n)
		mask = mask.expand(mask.size(0), x.size(1), mask.size(2))
		weight = self.determine_weight(sizes)
		x = x.unsqueeze(2) * weight * mask.float()
		if self.relaxed:
			x, _ = cont_sort(x, perm)
		else:
			x = x.scatter(2, perm, x)
		return x, mask

	def determine_weight(self, sizes):
		"""
			Piecewise linear function. Evaluates f at the ratios in sizes.
			This should be a faster implementation than doing the sum over max terms, since we know that most terms in it are 0.
		"""
		# share same sequence length within each sample, so copy weighht across batch dim
		weight = self.weight.unsqueeze(0)
		weight = weight.expand(sizes.size(0), weight.size(1), weight.size(2))
		# linspace [0, 1] -> linspace [0, n_pieces]
		index = self.n_pieces * sizes
		index = index.unsqueeze(1)
		index = index.expand(index.size(0), weight.size(1), index.size(2))
		# points in the weight vector to the left and right
		idx = index.long()
		frac = index.frac()
		left = weight.gather(2, idx)
		right = weight.gather(2, (idx + 1).clamp(max=self.n_pieces))
		# interpolate between left and right point
		return (1 - frac) * left + frac * right


def fill_sizes(sizes, x=None):
	"""
		sizes is a LongTensor of size [batch_size], containing the set sizes.
		Each set size n is turned into [0/(n-1), 1/(n-1), ..., (n-2)/(n-1), 1, 0, 0, ..., 0, 0].
		These are the ratios r at which f is evaluated at.
		The 0s at the end are there for padding to the largest n in the batch.
		If the input set x is passed in, it guarantees that the mask is the correct size even when sizes.max()
		is less than x.size(), which can be a case if there is at least one padding element in each set in the batch.
	"""
	if x is not None:
		max_size = x.size(2)
	else:
		max_size = sizes.max()
	size_tensor = sizes.new(sizes.size(0), max_size).float().fill_(-1)
	size_tensor = torch.arange(end=max_size, device=sizes.device, dtype=torch.float32)
	size_tensor = size_tensor.unsqueeze(0) / (sizes.float() - 1).clamp(min=1).unsqueeze(1)
	mask = size_tensor <= 1
	mask = mask.unsqueeze(1)
	return size_tensor.clamp(max=1), mask.float()


def deterministic_sort(s, tau):
	"""
	"Stochastic Optimization of Sorting Networks via Continuous Relaxations" https://openreview.net/forum?id=H1eSS3CcKX
	Aditya Grover, Eric Wang, Aaron Zweig, Stefano Ermon
	s: input elements to be sorted. Shape: batch_size x n x 1
	tau: temperature for relaxation. Scalar.
	"""
	n = s.size()[1]
	one = torch.ones((n, 1), dtype = torch.float32, device=s.device)
	A_s = torch.abs(s - s.permute(0, 2, 1))
	B = torch.matmul(A_s, torch.matmul(one, one.transpose(0, 1)))
	scaling = (n + 1 - 2 * (torch.arange(n, device=s.device) + 1)).type(torch.float32)
	C = torch.matmul(s, scaling.unsqueeze(0))
	P_max = (C - B).permute(0, 2, 1)
	sm = torch.nn.Softmax(-1)
	P_hat = sm(P_max / tau)
	return P_hat


def cont_sort(x, perm=None, temp=1):
	""" Helper function that calls deterministic_sort with the right shape.
	Since it assumes a shape of (batch_size, n, 1) while the input x is of shape (batch_size, channels, n),
	we can get this to the right shape by merging the first two dimensions.
	If an existing perm is passed in, we compute the "inverse" (transpose of perm) and just use that to unsort x.
	"""
	original_size = x.size()
	x = x.view(-1, x.size(2), 1)
	if perm is None:
		perm = deterministic_sort(x, temp)
	else:
		perm = perm.transpose(1, 2)
	x = perm.matmul(x)
	x = x.view(original_size)
	return x, perm


def hungarian_loss(predictions, targets):
	# predictions and targets shape :: (n, c, s)
	predictions, targets = outer(predictions, targets)
	# squared_error shape :: (n, s, s)
	squared_error = torch.nn.functional.smooth_l1_loss(predictions, targets, reduction="none").mean(1)

	squared_error_np = squared_error.detach().cpu().numpy()
	indices = [hungarian_loss_per_sample(squared_error_np[i]) for i in range(squared_error_np.shape[0])]
	losses = [
		sample[row_idx, col_idx].mean()
		for sample, (row_idx, col_idx) in zip(squared_error, indices)
	]
	total_loss = torch.mean(torch.stack(list(losses)))
	return total_loss


def hungarian_loss_per_sample(sample_np):
	return scipy.optimize.linear_sum_assignment(sample_np)


def outer(a, b=None):
	""" Compute outer product between a and b (or a and a if b is not specified). """
	if b is None:
		b = a
	size_a = tuple(a.size()) + (b.size()[-1],)
	size_b = tuple(b.size()) + (a.size()[-1],)
	a = a.unsqueeze(dim=-1).expand(*size_a)
	b = b.unsqueeze(dim=-2).expand(*size_b)
	return a, b


if __name__ == '__main__':

	dim = 3
	max_n = 5
	batch_size = 16

	# enc = Encoder(dim=dim)
	# dec = Decoder(encoder=enc, dim=dim, max_n=max_n)
	ae = AutoEncoder(dim=dim, hidden_dim=96, max_n=max_n)

	data_list = []
	batch_list = []
	for i in range(batch_size):
		n = torch.randint(low=1, high=max_n, size=(1,))
		x = torch.randn(n[0], dim)
		data_list.append(x)
		batch_list.append(torch.ones(n) * i)
	x = torch.cat(data_list, dim=0)
	batch = torch.cat(batch_list, dim=0).int()

	xr, batchr = ae(x, batch)

	print(x.shape, xr.shape)
	print(batch.shape, batchr.shape)

	loss = ae.loss()

	breakpoint()

