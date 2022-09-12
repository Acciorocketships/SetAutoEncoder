import torch
import plotly.graph_objects as go

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
			self.create_shape(colour=colour, pos=pos, radius=radius, alpha=alpha, line=line)

	def create_shape(self, colour, pos, radius, alpha=1, line={}):
		def rgb2hex(rgb, bounds=1):
			rgb = torch.clamp(rgb, 0, bounds) / bounds
			rgbint = (rgb * 255).int()
			return "#{:02x}{:02x}{:02x}".format(rgbint[0], rgbint[1], rgbint[2])
		self.fig.add_shape(type="circle",
					  x0=pos[0]-radius, y0=pos[1]-radius,
					  x1=pos[0]+radius, y1=pos[1]+radius,
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

if __name__ == "__main__":
	vis = Visualiser(visible=True)
	x = torch.randn(4,6)
	x_noisy = x + torch.randn(4,6) / 10
	vis.show_objects(x)
	vis.show_objects(x_noisy, alpha=0.5, line={"dash": "dot"})
	vis.render()