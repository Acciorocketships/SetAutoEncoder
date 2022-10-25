from matplotlib import pyplot
from matplotlib.patches import Rectangle, Circle
from sae.util import *
import torch

class Visualiser:

	def __init__(self, visible=False):
		self.visible = visible
		self.reset()

	def reset(self):
		self.fig, self.ax = pyplot.subplots(figsize=(10,10))


	def show_objects(self, x, agent_pos=torch.zeros(2), **kwargs):
		x = x.detach()
		for i in range(x.shape[0]):
			colour = torch.abs(x[i,:3])
			radius = torch.abs(x[i,3]) / 8
			pos = (x[i,4:6] - agent_pos) * 1
			self.create_shape(colour=colour, pos=pos, radius=radius, type="circle", **kwargs)


	def highlight_agent(self, agent_pos):
		self.create_shape(colour="blue", pos=agent_pos, radius=0.1, type="box")


	def show_agents(self, agent_idx, agent_pos, edge_idx):
		done = set()
		for idxi in agent_idx:
			connections = edge_idx[edge_idx[:,0]==idxi]
			for i in range(connections.shape[0]):
				edge1 = (connections[i, 0].item(), connections[i, 1].item())
				edge2 = (connections[i, 1].item(), connections[i, 0].item())
				if edge1 in done or edge1[0]==edge1[1]:
					continue
				else:
					done.add(edge1)
					done.add(edge2)
				pos1 = agent_pos[connections[i, 0],:]
				pos2 = agent_pos[connections[i, 1], :]
				self.create_shape(colour=torch.zeros(3), pos=pos1, pos2=pos2, type="line", linewidth=1.5, linestyle="dotted")
		for idxi in agent_idx:
			self.create_shape(colour=torch.zeros(3), pos=agent_pos[idxi], radius=0.1, type="box")

	def create_shape(self, colour, pos, radius=None, pos2=None, type="circle", **kwargs):
		def rgb2hex(rgb, bounds=1):
			rgb = torch.clamp(rgb, 0, bounds) / bounds
			rgbint = (rgb * 255).int()
			return "#{:02x}{:02x}{:02x}".format(rgbint[0], rgbint[1], rgbint[2])
		colour = rgb2hex(colour) if not isinstance(colour, str) else colour
		if pos2 is not None:
			if type == "line":
				self.ax.arrow(pos[0], pos[1], pos2[0]-pos[0], pos2[1]-pos[1], color=colour, **kwargs)
		elif radius is not None:
			if type == "circle":
				shape = Circle((pos[0], pos[1]), radius=radius, facecolor=colour, edgecolor="black", **kwargs)
			elif type == "box":
				shape = Rectangle((pos[0]-radius, pos[1]-radius), width=2*radius, height=2*radius, facecolor=colour, edgecolor="black", **kwargs)
			self.ax.add_patch(shape)

	def render(self, lim=None, grid=False, axis=False, margin=0.):
		self.ax.axis("equal")
		if isinstance(lim, list) and len(lim) == 4:
			self.ax.set_xlim(left=lim[0], right=lim[1])
			self.ax.set_ylim(bottom=lim[2], top=lim[3])
		elif isinstance(lim, list) and len(lim) == 2:
			self.ax.set_xlim(left=lim[0], right=lim[1])
			self.ax.set_ylim(bottom=lim[0], top=lim[1])
		elif isinstance(lim, float):
			self.ax.set_xlim(left=-lim, right=lim)
			self.ax.set_ylim(bottom=-lim, top=lim)
		self.fig.subplots_adjust(bottom=margin, top=1-margin, left=margin, right=1-margin)
		if not axis:
			for tick in self.ax.xaxis.get_major_ticks() + self.ax.yaxis.get_major_ticks():
				tick.tick1line.set_visible(False)
				tick.tick2line.set_visible(False)
				tick.label1.set_visible(False)
				tick.label2.set_visible(False)
		if grid:
			self.ax.grid()
			self.ax.set_axisbelow(True)
		if self.visible:
			self.fig.show()
		return self.fig

	def save(self, path):
		self.fig.savefig(path, bbox_inches="tight")

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
		obj_env = data["obj_all"][data["obj_env"][batch],:]
		agent_num = data["agent_env"][batch][agent]
		center = data["agent_pos"][agent_num]
		self.show_objects(x=obj_env, agent_pos=center, alpha=0.3, linestyle="dashed")
		self.show_objects(x=obj_data["obj_per_agent_pred"][agent_num], agent_pos=center)
		self.show_agents(agent_idx=data["agent_env"][batch], agent_pos=data["agent_pos"]-center, edge_idx=data["edge_idx"])
		self.highlight_agent(agent_pos=center-center)


if __name__ == "__main__":
	vis = Visualiser(visible=True)
	x = torch.randn(4,6)
	x_noisy = x + torch.randn(4,6) / 10
	vis.show_objects(x_noisy, alpha=0.3, linestyle="dashed")
	vis.show_objects(x)
	vis.render(grid=True)