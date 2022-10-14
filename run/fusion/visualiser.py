import plotly.graph_objects as go
from sae.util import *
import torch

class Visualiser:

	def __init__(self, visible=False):
		self.visible = visible
		self.reset()

	def reset(self):
		self.fig = go.Figure()
		self.fig.update_yaxes(
			scaleanchor="x",
			scaleratio=1,
		)

	def show_objects(self, x, agent_pos=torch.zeros(2), alpha=1, line={}):
		x = x.detach()
		for i in range(x.shape[0]):
			colour = torch.abs(x[i,:3])
			radius = torch.abs(x[i,3]) / 8
			pos = (x[i,4:6] - agent_pos) * 1
			self.create_shape(colour=colour, pos=pos, radius=radius, alpha=alpha, line=line, type="circle")

	def show_agents(self, agent_idx, agent_pos, edge_idx):
		for idxi in agent_idx:
			connections = edge_idx[edge_idx[:,0]==idxi]
			for i in range(connections.shape[0]):
				pos0 = agent_pos[connections[i, 0],:]
				pos1 = agent_pos[connections[i, 1], :]
				self.create_shape(colour=torch.zeros(3), pos=pos0, radius=pos1, type="line")
		for idxi in agent_idx:
			self.create_shape(colour=torch.zeros(3), pos=agent_pos[idxi], radius=0.1, type="rect")

	def create_shape(self, colour, pos, radius, alpha=1, line={}, type="circle"):
		def rgb2hex(rgb, bounds=1):
			rgb = torch.clamp(rgb, 0, bounds) / bounds
			rgbint = (rgb * 255).int()
			return "#{:02x}{:02x}{:02x}".format(rgbint[0], rgbint[1], rgbint[2])
		if not isinstance(radius, torch.Tensor) or radius.numel() < 2:
			self.fig.add_shape(type=type,
					  x0=pos[0]-radius, y0=pos[1]-radius,
					  x1=pos[0]+radius, y1=pos[1]+radius,
					  opacity=alpha,
					  fillcolor=rgb2hex(colour),
					  line=go.layout.shape.Line(**line),
					  )
		else:
			self.fig.add_shape(type=type,
					   x0=pos[0], y0=pos[1],
					   x1=radius[0], y1=radius[1],
					   opacity=alpha,
					   fillcolor=rgb2hex(colour),
					   line=go.layout.shape.Line(**line),
					   )

	def render(self, lim=[-2, 2]):
		if len(lim) == 4:
			self.fig.update_layout(xaxis_range=[lim[0], lim[1]])
			self.fig.update_layout(yaxis_range=[lim[2], lim[3]])
		elif len(lim) == 2:
			self.fig.update_layout(xaxis_range=[lim[0], lim[1]])
			self.fig.update_layout(yaxis_range=[lim[0], lim[1]])
		if self.visible:
			self.fig.show()
		return self.fig

	def get_data(self, model, data, layer=0):
		objs_input = model.ae_train_data[layer]['x']
		objs_pred = model.ae_train_data[layer]['xr']
		num_objs_per_agent_in = model.ae_train_data[layer]['n']
		num_objs_per_agent_pred = model.ae_train_data[layer]['n_pred']
		obj_per_agent_in = create_nested(objs_input, num_objs_per_agent_in)
		obj_per_agent_pred = create_nested(objs_pred, num_objs_per_agent_pred)
		obj_idx_per_agent = model.model_inputs[layer]['obj_idx']
		obj_per_agent_true = index_with_nested(data['obj_all'], obj_idx_per_agent, dim=0)
		return {
			"obj_per_agent_in": obj_per_agent_in,
			"obj_per_agent_pred": obj_per_agent_pred,
			"obj_per_agent_true": obj_per_agent_true,
		}

	def visualise_obs(self, model, data, batch=0, agent=0, layer=0):
		self.reset()
		if len(data["agent_env"][batch]) <= agent:
			return None
		obj_data = self.get_data(model=model, data=data, layer=layer)
		agent_num = data["agent_env"][batch][agent]
		center = data["agent_pos"][agent_num]
		self.show_agents(agent_idx=data["agent_env"][batch], agent_pos=data["agent_pos"]-center, edge_idx=data["edge_idx"])
		self.show_objects(x=obj_data["obj_per_agent_in"][agent_num], agent_pos=center, alpha=0.3, line={"dash": "dot"})
		self.show_objects(x=obj_data["obj_per_agent_pred"][agent_num], agent_pos=center)

if __name__ == "__main__":
	vis = Visualiser(visible=True)
	x = torch.randn(4,6)
	x_noisy = x + torch.randn(4,6) / 10
	vis.show_objects(x)
	vis.show_objects(x_noisy, alpha=0.5, line={"dash": "dot"})
	vis.render()