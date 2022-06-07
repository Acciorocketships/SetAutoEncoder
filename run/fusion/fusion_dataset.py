import numpy as np
import torch
from torch_geometric import utils
import networkx as nx
from torch_cluster import radius, radius_graph
from torch_geometric.data import HeteroData, Data, Batch

class ObsEnv:

    def __init__(self,
                 mean_agents=8,
                 mean_objects=10,
                 obs_range=0.3,
                 com_range=0.3,
                 features=4,
                 ):
        self.mean_agents = mean_agents
        self.mean_objects = mean_objects
        self.obs_range = obs_range
        self.com_range = com_range
        self.features = features

    def sample(self):
        # Sample
        n_agents = torch.poisson(torch.tensor(self.mean_agents).float()).int().item()
        n_objects = torch.poisson(torch.tensor(self.mean_objects).float()).int().item()
        agent_pos = torch.rand(n_agents, 2)
        object_pos = torch.rand(n_objects, 2)
        # Filter Unseen Objects
        obs_dist = ((agent_pos.unsqueeze(1) - object_pos.unsqueeze(0)) ** 2).sum(dim=-1) ** 0.5
        num_seen_by = (obs_dist < self.obs_range).sum(dim=0) # number of agents that see each object (size: n_objects)
        object_pos = object_pos[num_seen_by>0]
        n_objects = object_pos.shape[0]
        # Creat Graph
        object_features = torch.rand(n_objects, self.features)
        agent_edges = radius_graph(agent_pos, r=self.com_range, loop=True)
        obs_edges = radius(object_pos, agent_pos, r=self.obs_range)
        data = HeteroData({
            "agent": {
                "pos": agent_pos,
                "edge_index": agent_edges,
            },
            "object": {
                "pos": object_pos,
                "x": object_features,
            },
            ("agent", "observe", "object"): {
                "edge_index": obs_edges,
            }
        })
        # Ensure Connected Graph
        g = utils.to_networkx(Data(x=data['agent'].pos, edge_index=data['agent'].edge_index), to_undirected=True)
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
            updated_edges = torch.cat([data['agent'].edge_index, new_edges.T], dim=1)
            data['agent'].edge_index = updated_edges
            g = utils.to_networkx(Data(x=data['agent'].pos, edge_index=data['agent'].edge_index), to_undirected=True)
            conn = list(nx.connected_components(g))
        return data

    def sample_n(self, n):
        datas = [self.sample() for _ in range(n)]
        return Batch.from_data_list(datas)


if __name__ == '__main__':
    env = ObsEnv()
    data = env.sample_n(10)
    breakpoint()
