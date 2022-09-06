import torch
from typing import Optional

def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1, dim_size: Optional[int] = None) -> torch.Tensor:
	index = broadcast(index, src, dim)
	size = list(src.size())
	if dim_size is not None:
		size[dim] = dim_size
	elif index.numel() == 0:
		size[dim] = 0
	else:
		size[dim] = int(index.max()) + 1
	out = torch.zeros(size, dtype=src.dtype, device=src.device)
	return out.scatter_add_(dim, index.long(), src)

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
	if dim < 0:
		dim = other.dim() + dim
	if src.dim() == 1:
		for _ in range(0, dim):
			src = src.unsqueeze(0)
	for _ in range(src.dim(), other.dim()):
		src = src.unsqueeze(-1)
	src = src.expand_as(other)
	return src

def size_nested(input, dim):
	if dim == 0:
		size = torch.tensor(len(input._nested_tensor_size()))
	else:
		dim = dim - 1 if dim > 0 else dim
		size = torch.tensor([size[dim] for size in input._nested_tensor_size()])
		if torch.all(size==size[0]):
			size = size[0]
	return size

def shape_nested(input):
	if input.is_nested:
		shape = []
		num_dims = input.dim()
		for dim in range(num_dims):
			dim_size = size_nested(input, dim=dim)
			if len(dim_size.shape) > 0:
				dim_size = None
			else:
				dim_size = dim_size.item()
			shape.append(dim_size)
		return tuple(shape)
	else:
		return tuple(input.shape)

def nested_to_batch(nested, return_sizes=False):
	tensor_list = nested.unbind()
	flat = torch.cat(tensor_list, dim=0)
	sizes = size_nested(nested, dim=1)
	if return_sizes:
		return flat, sizes
	else:
		batch = torch.arange(len(tensor_list)).repeat_interleave(sizes)
		return flat, batch

def permute_nested(nt, perm):
	x_flat, sizes = nested_to_batch(nt, return_sizes=True)
	x_perm = x_flat[perm]
	return create_nested(x_perm, sizes)

def create_nested(x, sizes):
	return torch.nested_tensor(torch.split(x, sizes.tolist()))

def create_nested_batch(x, batch, dim_size=None):
	if dim_size is None:
		dim_size = torch.max(batch)+1
	sizes = torch.zeros(dim_size).scatter_(dim=0, index=batch, src=torch.ones(batch.shape[0]), reduce='add').int()
	return create_nested(x, sizes)

def truncate_nested(nt, sizes, dim=1):
	dim = dim-1
	size_update = lambda size, dimsize: tuple(list(size)[:dim] + [dimsize] + list(size)[dim+1:])
	return torch.nested_tensor([
			torch.cat([
				torch.narrow(xi, dim=dim, start=0, length=min(size, xi.shape[dim])),
				torch.full(size=size_update(xi.shape, max(size-xi.shape[dim], 0)), fill_value=torch.nan)
			], dim=dim)
		for (xi, size) in zip(nt.unbind(), sizes)])