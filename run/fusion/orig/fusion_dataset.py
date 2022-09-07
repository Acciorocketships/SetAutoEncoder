import numpy as np
import torch
from torch_geometric import utils
import networkx as nx
from torch_cluster import radius, radius_graph
from torch_geometric.data import HeteroData, Data, Batch
from graphtorch.util import *

class ObsEnv:

    def __init__(self,
                 mean_agents = 8,
                 mean_objects = 10,
                 obs_range = 0.3,
                 com_range = 0.3,
                 feat_dim = 4,
                 pos_dim = 2,
                 **kwargs,
                 ):
        self.mean_agents = mean_agents
        self.mean_objects = mean_objects
        self.obs_range = obs_range
        self.com_range = com_range
        self.feat_dim = feat_dim
        self.pos_dim = pos_dim

    def sample(self):
        # Sample
        n_agents = 0
        n_objects = 0
        while n_agents == 0 or n_agents > 2*self.mean_agents:
            n_agents = torch.poisson(torch.tensor(self.mean_agents).float()).int().item()
        agent_pos = torch.rand(n_agents, self.pos_dim)
        while n_objects == 0 or n_objects > 2*self.mean_objects:
            n_objects = torch.poisson(torch.tensor(self.mean_objects).float()).int().item()
            object_pos = torch.rand(n_objects, self.pos_dim)
            # Filter Unseen Objects
            obs_dist = ((agent_pos.unsqueeze(1) - object_pos.unsqueeze(0)) ** 2).sum(dim=-1) ** 0.5
            num_seen_by = (obs_dist < self.obs_range).sum(dim=0) # number of agents that see each object (size: n_objects)
            object_pos = object_pos[num_seen_by>0]
            n_objects = object_pos.shape[0]
        # Creat Graph
        object_features = torch.rand(n_objects, self.feat_dim)
        agent_edges = radius_graph(agent_pos, r=self.com_range, loop=True)
        obs_edges = radius(object_pos, agent_pos, r=self.obs_range)
        data = HeteroData({
            "agent": {
                "pos": agent_pos,
            },
            "object": {
                "pos": object_pos,
                "x": object_features,
            },
            ("agent", "observe", "object"): {
                "edge_index": obs_edges,
            },
            ("agent", "communicate", "agent"): {
                "edge_index": agent_edges,
            }
        })
        # Ensure Connected Graph
        g = utils.to_networkx(Data(x=data['agent'].pos, edge_index=data[("agent", "communicate", "agent")].edge_index), to_undirected=True)
        conn = list(nx.connected_components(g))
        com_range = self.com_range
        while len(conn) > 1:
            posi = data['agent'].pos.unsqueeze(1)
            posj = data['agent'].pos.unsqueeze(0)
            dist = ((posi - posj) ** 2).sum(dim=-1) ** 0.5
            dist[dist <= com_range] = np.Inf
            num_new_edges = 2 if len(conn)==2 else 4*(len(conn)-1)
            new_edges_data = torch.topk(dist.view(-1), k=num_new_edges, largest=False)
            new_edges_flat = new_edges_data[1]
            com_range = torch.max(new_edges_data[0])
            new_edges = torch.tensor([(edge//dist.shape[0], edge%dist.shape[0]) for edge in new_edges_flat])
            updated_edges = torch.cat([data[("agent", "communicate", "agent")].edge_index, new_edges.T], dim=1)
            data[("agent", "communicate", "agent")].edge_index = updated_edges
            g = utils.to_networkx(Data(x=data['agent'].pos, edge_index=data[("agent", "communicate", "agent")].edge_index), to_undirected=True)
            conn = list(nx.connected_components(g))
        return data

    def sample_nested(self):
        data = self.sample()
        edge_idx_obj = data[("agent", "observe", "object")].edge_index.T
        edge_idx_agent = data[("agent", "communicate", "agent")].edge_index.T
        obj_flat = data["object"].x
        agent_pos = data["agent"].pos
        obj = scatter_nested(obj_flat, idxi=edge_idx_obj[:,0], idxj=edge_idx_obj[:,1])
        obj_idx = scatter_nested(edge_idx_obj[:,1], idxi=edge_idx_obj[:,0], idxj=torch.arange(edge_idx_obj.shape[0]))
        return {
            "obj_all": obj_flat,
            "agent_pos": agent_pos,
            "obj": obj,
            "obj_idx": obj_idx,
            "edge_idx": edge_idx_agent,
        }

    def sample_n(self, n):
        datas = [self.sample() for _ in range(n)]
        return Batch.from_data_list(datas)

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



if __name__ == '__main__':
    env = ObsEnv()
    # data = env.sample_n(1000)
    data = env.sample_n_nested_combined(10)
    breakpoint()
