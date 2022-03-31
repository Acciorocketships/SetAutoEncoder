import torch
from torch import nn
import copy
import types
from functools import partial


class LinearHyper(nn.Module):

	def __init__(self, input_dim, output_dim, params, bias=True):
		super().__init__()
		self.in_features = input_dim
		self.out_features = output_dim
		self.batch = params.shape[0]
		self.include_bias = bias
		self.weight = params[:,:input_dim*output_dim].reshape(self.batch, input_dim, output_dim)
		self.num_params = input_dim*output_dim
		if self.include_bias:
			self.bias = params[:,output_dim*input_dim:output_dim*input_dim+output_dim].reshape(self.batch, 1, output_dim)
			self.num_params += output_dim


	def __repr__(self):
		return "LinearHyper(in_features={infeat}, out_features={outfeat}, batch={batch}, bias={bias})".format(infeat=self.in_features, outfeat=self.out_features, batch=self.batch, bias=self.include_bias)


	def forward(self, x):
		if len(x.shape) == 2:
			# x: batch x in_dim
			y = torch.bmm(x.unsqueeze(1), self.weight)
			if self.include_bias:
				y += self.bias
			return y[:,0,:]
		elif len(x.shape) == 3:
			# x: batch x N x in_dim
			y = torch.bmm(x, self.weight)
			if self.include_bias:
				y += self.bias
			return y
		else:
			raise ValueError("Input shape of {shape} not valid in LinearHyper".format(x.shape))



class Elementwise(nn.Module):

	def __init__(self, dim, params=None, bias=True):
		super().__init__()
		self.features = dim
		self.include_bias = bias
		self.num_params = dim+1 if self.include_bias else dim
		self.batch = None
		self.weight = None
		self.bias = None
		if params is not None:
			self.batch = params.shape[0]
			self.set_params(params)


	def __repr__(self):
		return "Elementwise(features={feat}, batch={batch}, bias={bias})".format(feat=self.features, batch=self.batch, bias=self.include_bias)


	def set_params(self, params):
		self.batch = params.shape[0]
		self.weight = params[:,:self.features].reshape(self.batch, self.features)
		if self.include_bias:
			self.bias = params[:,self.features:self.num_params].reshape(self.batch, 1)


	def forward(self, x):
		if len(x.shape) == 2:
			# x: batch x in_dim
			y = x * self.weight
			if self.include_bias:
				y += self.bias
			return y
		elif len(x.shape) == 3:
			# x: batch x N x in_dim
			y = x.unsqueeze(1) * self.weight
			if self.include_bias:
				y += self.bias
			return y
		else:
			raise ValueError("Input shape of {shape} not valid in LinearHyper".format(x.shape))


def replace_hyper_linear(module, params):
	input_dim = module.in_features
	output_dim = module.out_features
	bias = (module.bias != None)
	linear = LinearHyper(input_dim=input_dim, output_dim=output_dim, params=params, bias=bias)
	return linear


def replace_hyper_elementwise(module, params):
	module.set_params(params)
	return module


def update_module_params(module, params=None, i=0, param_dim=-1,
							filter_cond=lambda module: isinstance(module, nn.Linear) or isinstance(module, LinearHyper), 
							replace_func=replace_hyper_linear):
	# params: batch x n_params
	for name, submodule in module._modules.items():
		if any(submodule.children()):
			update_module_params(submodule, params=params, filter_cond=filter_cond, replace_func=replace_func, i=i, param_dim=param_dim)
		elif filter_cond(submodule):
			new_param = None
			if params is not None:
				num_params = get_num_params(submodule)
				new_param = select_index(params, dim=param_dim, idx=slice(i,i+num_params))
				i += num_params
			module._modules[name] = replace_func(submodule, params=new_param)


def get_num_params(module):
	if hasattr(module, "num_params"):
		return module.num_params
	children = list(module.children())
	if len(children) > 0:
		child_params = 0
		for child in children:
			child_params += get_num_params(child)
		return child_params
	return sum(param.numel() for param in module.parameters())


def select_index(arr, dim, idx):
	idx_list = [slice(None)] * dim + [idx]
	return arr.__getitem__(idx_list)


if __name__ == '__main__':

	f = nn.Sequential(nn.Linear(2,4), nn.ReLU(), nn.Linear(4, 3))
	params = torch.randn(16, get_num_params(f))
	params.requires_grad = True
	x = torch.ones(16,2)

	update_module_params(f, params, replace_func=replace_hyper_linear)

	y = f(x)

	g = torch.autograd.grad(y.mean(), params)

	import pdb; pdb.set_trace()