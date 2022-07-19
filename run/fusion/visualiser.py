from vmas.simulator.rendering import Viewer, Transform
from vmas.simulator.core import Sphere
import torch

class Visualiser:

	def __init__(self, visible=False):
		self.viewer = Viewer(width=800, height=800, visible=visible)
		self.viewer.set_bounds(-1., 1., -1., 1.)
		self.caption = None

	def show_objects(self, x, agent_pos=torch.zeros(2), alpha=1):
		for i in range(x.shape[0]):
			colour = torch.abs(x[i,:3])
			radius = torch.abs(x[i,3]) / 8
			pos = (x[i,4:6] - agent_pos) * 1
			shape = self.create_shape(colour.tolist(), pos.tolist(), radius, alpha=alpha)
			self.viewer.add_geom(shape)

	def create_shape(self, colour, pos, radius, alpha=1):
		shape = Sphere(radius=radius).get_geometry()
		xform = Transform()
		xform.set_translation(*pos)
		shape.add_attr(xform)
		shape.set_color(*colour, alpha)
		return shape

	def get_data(self, model, agent=0, layer=0):
		if layer == model.gnn_nlayers:
			x_true = model.merge_gnn.get_values("x_output")[-1]
			batch_true = model.merge_gnn.get_values("batch_output")[-1]
			perm = model.merge_gnn.get_values("perm_output")[-1]
			x_pred = model.decoder.get_x_pred()
			batch_pred = model.decoder.get_batch_pred()
		else:
			x_true = model.merge_gnn.get_values("x_output")[layer]
			batch_true = model.merge_gnn.get_values("batch_output")[layer]
			perm = model.merge_gnn.get_values("perm_output")[layer]
			x_pred = model.merge_gnn.get_values("x_pred")[layer]
			batch_pred = model.merge_gnn.get_values("batch_pred")[layer]
		agent_pos = model.encode_gnn.agent_pos[agent]
		x_true = x_true[perm]
		x_true_subset = x_true[batch_true==agent]
		x_pred_subset = x_pred[batch_pred==agent]
		return x_true_subset, x_pred_subset, agent_pos

	def visualise_obs(self, model, agent=0, layer=0, true=True, pred=False):
		self.viewer.geoms = []
		true_alpha = 0.3 if pred else 1.
		x_true_subset, x_pred_subset, agent_pos = self.get_data(model, agent=agent, layer=layer)
		if model.position == "rel":
			agent_pos = torch.zeros(2)
		if true:
			self.show_objects(x_true_subset.detach(), agent_pos, alpha=true_alpha)
			self.caption = self.create_caption(x_true_subset)
		if pred:
			self.show_objects(x_pred_subset.detach(), agent_pos, alpha=1.)
			self.caption = self.create_caption(x_pred_subset)

	def visualise_map(self, data, agent=0):
		self.viewer.geoms = []
		batch_num = data['agent'].batch[agent]
		datai = data[batch_num]
		obj_feat = datai['object'].x
		obj_pos = datai['object'].pos
		agent_pos = data['agent'].pos[agent]
		x = torch.cat([obj_feat, obj_pos], dim=1)
		self.show_objects(x, agent_pos=agent_pos)
		self.caption = self.create_caption(x)

	def create_caption(self, x, agent_pos=None):
		torch.set_printoptions(precision=2)
		caption = ""
		if agent_pos is not None:
			caption += "agent pos:\n"
			caption += str(agent_pos.detach())[7:-1].replace(" ", "").replace("\n", "\n ")
			caption += "\nx:\n"
		caption += str(x.detach())[7:-1].replace(" ", "").replace("\n", "\n ")
		return caption

	def render(self):
		return self.viewer.render(return_rgb_array=True)