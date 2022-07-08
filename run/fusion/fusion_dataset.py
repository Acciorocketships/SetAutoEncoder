import numpy as np
import torch
from torch_geometric import utils
import networkx as nx
from torch_cluster import radius, radius_graph
from torch_geometric.data import HeteroData, Data, Batch

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

    def sample_n(self, n):
        datas = [self.sample() for _ in range(n)]
        return Batch.from_data_list(datas)


if __name__ == '__main__':
    env = ObsEnv()
    data = env.sample_n(1000)
