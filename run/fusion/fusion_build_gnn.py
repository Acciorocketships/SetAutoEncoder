from torch_geometric.nn import Sequential, JumpingKnowledge
from torch_geometric.nn import EdgeConv, GraphConv
import torch
from torch import nn

def build_gnn(gnn_type=EdgeConv, gnn_kwargs=None, layers=[8,8], skip_layers=None, skip_connections="first", activation=nn.ReLU, **kwargs):

	gnn_layers = []

	def output_name(layer):
		return "x{layer}".format(layer=layer)
	def input_name(layer):
		return output_name(layer) + "_in"
	def input_string_all(layer):
		string = "x, "
		for i in range(1, layer):
			string += output_name(i)
		if string[-1] == ",":
			string = string[:-1]
		return string + " -> {input}".format(input=input_name(layer))
	def input_string_last(layer):
		return "{last_output} -> {input}".format(last_output=output_name(layer - 1), input=input_name(layer))

	l = 1
	n_layers = len(layers) - 1
	for in_dim, out_dim in zip(layers[:-1], layers[1:]):
		input_dim_cat = in_dim
		if (l > 1) and (skip_layers == "all" or (skip_layers == "last" and l == n_layers)):
			if skip_connections == "all":
				input_dim_cat = sum(layers[:l])
				gnn_layers.append((lambda *x: list(x), input_string_all(l)))
			elif skip_connections == "first":
				input_dim_cat = in_dim + layers[0]
				gnn_layers.append((lambda *x: list(x), "x, " + input_string_last(l)))
			else:
				raise ValueError("skip_connections must be 'all' or 'first', received {val}".format(val=skip_connections))
			gnn_layers.append((JumpingKnowledge("cat"), "{input} -> {input}".format(input=input_name(l))))
		else:
			if l == 1:
				gnn_layers.append((lambda x: x, "x -> x1_in"))
			else:
				gnn_layers.append((lambda x: x, input_string_last(l)))

		if gnn_kwargs is None:
			gnn_args = {"in_channels": input_dim_cat, "out_channels": out_dim}
		else:
			gnn_args = gnn_kwargs(in_channels=input_dim_cat, out_channels=out_dim)
		gnn_layer = gnn_type(**gnn_args)
		gnn_layers.append((gnn_layer, '{input}, edge_index, *args, **kwargs -> {output}'.format(input=input_name(l), output=output_name(l))))

		if l < n_layers:
			gnn_layers.append(activation())

		l += 1

	gnn = Sequential('x, edge_index, *args, **kwargs', gnn_layers)
	return gnn

if __name__ == '__main__':
	from torch_geometric.data import Data

	config = lambda in_channels, out_channels: {"nn": nn.Linear(2*in_channels, out_channels)}

	gnn = build_gnn(gnn_type=EdgeConv, gnn_kwargs=config, layers=[1,3,2], skip_layers="all", skip_connections="all")

	edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
	x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
	data = Data(x=x, edge_index=edge_index)

	out = gnn(x=data.x, edge_index=data.edge_index)

	print(out)