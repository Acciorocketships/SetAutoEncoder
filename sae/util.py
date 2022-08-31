import torch

def collect(input, index, dim_along=0, dim_select=1):
	'''
	input is the source tensor, of any shape
	index is a 1D tensor of shape input.shape[dim_along]
	dim_along is the dimension where each element you would like a different value
	dim_select is the dimension that elements in index are indexing

	For example, if you have input: (B, T, D) and index: (B,) where each element is in range(0, T),
	you can select the t-th element from each row in the batch dimension to produce an output of shape (B, D) with the following:
	output = collect(input, index, along_dim=0, dim_select=1)
	because we are indexing along dimension 0 (the B dim), and each element in index picks out an element in dimension 1 (the T dim).
	'''
	shape = list(input.shape)
	shape[dim_along] = index.shape[0]
	shape[dim_select] = 1
	unsqueeze_shape = [1] * len(shape)
	unsqueeze_shape[dim_along] = index.shape[0]
	index_unsqueezed = index.view(*unsqueeze_shape)
	index_expanded = index_unsqueezed.expand(*shape)
	out = torch.gather(input=input, index=index_expanded, dim=dim_select)
	return out.squeeze(dim_select)

if __name__ == "__main__":
	dim_along = 0
	dim_select = 1
	x = torch.rand(3,4,5)
	idx = torch.randint(0,x.shape[dim_select],(x.shape[dim_along],))
	y = collect(x, idx, dim_along=dim_along, dim_select=dim_select)
	breakpoint()