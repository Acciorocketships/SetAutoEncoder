import numpy as np
import torch
import networkx as nx
import scipy
from graphtorch.util import *

class ObsEnv:

    def __init__(self,
                 mean_agents = 8,
                 mean_objects = 10,
                 obs_range = 0.3,
                 com_range = 0.3,
                 feat_dim = 4,
                 pos_dim = 2,
                 concat_pos = True,
                 **kwargs,
                 ):
        self.mean_agents = mean_agents
        self.mean_objects = mean_objects
        self.obs_range = obs_range
        self.com_range = com_range
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim
        self.concat_pos = concat_pos

    def sample(self):
        # Sample
        n_agents = 0
        n_objects = 0
        while n_agents == 0 or n_agents > 2*self.mean_agents:
            n_agents = torch.poisson(torch.tensor(self.mean_agents).float()).int().item()
        agent_pos = torch.randn(n_agents, self.pos_dim)
        while n_objects == 0 or n_objects > 2*self.mean_objects:
            n_objects = torch.poisson(torch.tensor(self.mean_objects).float()).int().item()
            object_pos = torch.randn(n_objects, self.pos_dim)
            # Filter Unseen Objects
            obs_dist = ((agent_pos.unsqueeze(1) - object_pos.unsqueeze(0)) ** 2).sum(dim=-1) ** 0.5
            num_seen_by = (obs_dist < self.obs_range).sum(dim=0) # number of agents that see each object (size: n_objects)
            object_pos = object_pos[num_seen_by>0]
            n_objects = object_pos.shape[0]
        # Creat Graph
        object_features = torch.randn(n_objects, self.feat_dim)
        agent_edges = radius_graph(agent_pos, r=self.com_range, loop=True)
        obs_edges = radius(object_pos, agent_pos, r=self.obs_range)
        data = {
            "agent": {
                "pos": agent_pos,
            },
            "object": {
                "pos": object_pos,
                "x": torch.cat([object_features, object_pos], dim=-1) if (self.concat_pos) else object_features,
            },
            ("agent", "observe", "object"): {
                "edge_index": obs_edges,
            },
            ("agent", "communicate", "agent"): {
                "edge_index": agent_edges,
            }
        }
        # Ensure Connected Graph
        g = to_networkx(node_attrs={"pos": data['agent']["pos"]}, edge_index=data[("agent", "communicate", "agent")]["edge_index"], num_nodes=data['agent']["pos"].shape[0])
        conn = list(nx.connected_components(g))
        com_range = self.com_range
        while len(conn) > 1:
            posi = data['agent']["pos"].unsqueeze(1)
            posj = data['agent']["pos"].unsqueeze(0)
            dist = ((posi - posj) ** 2).sum(dim=-1) ** 0.5
            dist[dist <= com_range] = np.Inf
            num_new_edges = 2 if len(conn)==2 else 4*(len(conn)-1)
            new_edges_data = torch.topk(dist.view(-1), k=num_new_edges, largest=False)
            new_edges_flat = new_edges_data[1]
            com_range = torch.max(new_edges_data[0])
            new_edges = torch.tensor([(edge//dist.shape[0], edge%dist.shape[0]) for edge in new_edges_flat])
            updated_edges = torch.cat([data[("agent", "communicate", "agent")]["edge_index"], new_edges.T], dim=1)
            data[("agent", "communicate", "agent")]["edge_index"] = updated_edges
            g = to_networkx(node_attrs={"pos": data['agent']["pos"]}, edge_index=data[("agent", "communicate", "agent")]["edge_index"], num_nodes=data['agent']["pos"].shape[0])
            conn = list(nx.connected_components(g))
        return data

    def sample_nested(self):
        data = self.sample()
        edge_idx_obj = data[("agent", "observe", "object")]["edge_index"].T
        edge_idx_agent = data[("agent", "communicate", "agent")]["edge_index"].T
        obj_flat = data["object"]["x"]
        agent_pos = data["agent"]["pos"]
        obj = scatter_nested(obj_flat, idxi=edge_idx_obj[:,0], idxj=edge_idx_obj[:,1], size=agent_pos.shape[0])
        obj_idx = scatter_nested(edge_idx_obj[:,1], idxi=edge_idx_obj[:,0], idxj=torch.arange(edge_idx_obj.shape[0]), size=agent_pos.shape[0])
        return {
            "obj_all": obj_flat,
            "agent_pos": agent_pos,
            "obj": obj,
            "obj_idx": obj_idx,
            "edge_idx": edge_idx_agent,
        }

    def sample_n_nested(self, n):
        datas = [self.sample_nested() for _ in range(n)]
        keys = datas[0].keys()
        return {key: [datas[i][key] for i in range(len(datas))] for key in keys}

    def sample_n_nested_combined(self, n):
        datas = [self.sample_nested() for _ in range(n)]
        num_agents = torch.tensor([size_nested(data["obj"], dim=0) for data in datas])
        cum_num_agents = torch.cat([torch.zeros(1), torch.cumsum(num_agents, dim=0)[:-1]], dim=0).int()
        num_objs = torch.tensor([data["obj_all"].shape[0] for data in datas])
        cum_num_objs = torch.cat([torch.zeros(1), torch.cumsum(num_objs, dim=0)[:-1]], dim=0).int()
        agent_pos = torch.cat([data["agent_pos"] for data in datas], dim=0)
        edge_idxs = torch.cat([datas[i]["edge_idx"] + cum_num_agents[i] for i in range(len(datas))], dim=0)
        batch = torch.arange(len(datas)).repeat_interleave(num_agents)
        obj_all = torch.cat([data["obj_all"] for data in datas], dim=0)
        obj_list = sum([list(data["obj"].unbind()) for data in datas], [])
        obj = torch.nested_tensor(obj_list)
        obj_idx_list = sum([list(data["obj_idx"].unbind()) for data in datas], [])
        cum_num_objs_rep = cum_num_objs.repeat_interleave(num_agents)
        obj_idx_list = list(map(lambda x: x[0] + x[1], zip(obj_idx_list, cum_num_objs_rep)))
        obj_idx = torch.nested_tensor(obj_idx_list)
        return {
            "obj_all": obj_all,
            "agent_pos": agent_pos,
            "obj": obj,
            "obj_idx": obj_idx,
            "edge_idx": edge_idxs,
            "batch": batch,
        }


def to_networkx(edge_index, node_attrs, num_nodes):
    G = nx.Graph()

    G.add_nodes_from(range(num_nodes))

    for i, (u, v) in enumerate(edge_index.t().tolist()):
        G.add_edge(u, v)

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: node_attrs[key].tolist()[i]})
    return G


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=32):
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    if x.is_cuda:
        return radius(x, y, r, batch_x, batch_y, max_num_neighbors)

    x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x.detach().numpy())
    _, col = tree.query(
        y.detach().numpy(), k=max_num_neighbors, distance_upper_bound=r + 1e-8)
    col = [torch.from_numpy(c).to(torch.long) for c in col]
    row = [torch.full_like(c, i) for i, c in enumerate(col)]
    row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)
    mask = col < int(tree.n)
    return torch.stack([row[mask], col[mask]], dim=0)



def radius_graph(x,
                 r,
                 batch=None,
                 loop=False,
                 max_num_neighbors=32,
                 flow='source_to_target'):
    assert flow in ['source_to_target', 'target_to_source']
    row, col = radius(x, x, r, batch, batch, max_num_neighbors + 1)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)

if __name__ == '__main__':
    env = ObsEnv()
    # data = env.sample_n(1000)
    data = env.sample_n_nested_combined(10)
    breakpoint()

