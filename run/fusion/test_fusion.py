import unittest
from fusion_dataset import ObsEnv
from fusion_model import FusionModel
from fusion_gnn import MergeGNN, EncodeGNN
import torch
from torch_geometric.data import HeteroData, Data, Batch

class TestFusion(unittest.TestCase):
    # test agent to edge works
    # test filter works
    # test communication works
    # test output is full map

    @classmethod
    def setUpClass(cls):
        cls.feat_dim = 4
        cls.hidden_dim = 16
        cls.mean_agents = 4
        cls.mean_obj = 5
        cls.error4_data(cls)


    def random_data(self):
        env = ObsEnv(mean_agents=self.mean_agents, mean_objects=self.mean_obj, feat_dim=self.feat_dim)
        self.data = env.sample_n(1)

        self.obj_x = self.data['object'].x
        self.obj_pos = self.data['object'].pos
        self.agent_pos = self.data['agent'].pos
        self.obj_agent_edge_index = self.data[('agent', 'observe', 'object')].edge_index  # [agent_idx, obj_idx]
        self.agent_edge_index = self.data[('agent', 'communicate', 'agent')].edge_index

        self.x_idx = self.obj_agent_edge_index[0, :]
        self.obj_idx = self.obj_agent_edge_index[1, :]



    def error1_data(self):
        self.agent_edge_index = torch.tensor([[0, 2, 1, 0, 2, 2, 1],[0, 0, 1, 2, 2, 1, 2]])
        self.obj_agent_edge_index = torch.tensor([[0],[0]])
        self.obj_x = torch.tensor([[0.9562, 0.0688, 0.4053, 0.6451]])
        self.obj_pos = torch.tensor([[0.2275, 0.1838]])
        self.agent_pos = torch.tensor([[0.4576, 0.3272],[0.0575, 0.9048],[0.4501, 0.4772]])
        data = HeteroData({
            "agent": {
                "pos": self.agent_pos,
            },
            "object": {
                "pos": self.obj_pos,
                "x": self.obj_x,
            },
            ("agent", "observe", "object"): {
                "edge_index": self.obj_agent_edge_index,
            },
            ("agent", "communicate", "agent"): {
                "edge_index": self.agent_edge_index,
            }
        })
        self.data = Batch.from_data_list([data])
        self.x_idx = self.obj_agent_edge_index[0, :]
        self.obj_idx = self.obj_agent_edge_index[1, :]


    def error2_data(self):
        self.agent_edge_index = torch.tensor([[0, 5, 1, 2, 1, 2, 5, 3, 4, 0, 2, 5, 1, 5, 0, 2, 3, 1, 1, 0, 2, 3, 1, 4],
                                              [0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 5, 5, 5, 1, 2, 0, 1, 3, 0, 1, 3, 2, 4, 1]])
        self.obj_agent_edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 5, 5, 5, 5],
                                                  [0, 1, 4, 3, 4, 0, 1, 3, 4, 2, 0, 1, 3, 4]])
        self.obj_x =  torch.tensor([[0.1739, 0.1655, 0.3468, 0.2432],
                                    [0.6342, 0.5083, 0.7763, 0.3976],
                                    [0.1338, 0.6652, 0.8248, 0.8279],
                                    [0.9121, 0.2734, 0.4609, 0.6362],
                                    [0.4090, 0.6820, 0.6805, 0.4833]])
        self.obj_pos = torch.tensor([[0.1184, 0.7796],
                                     [0.0272, 0.7945],
                                     [0.0677, 0.0017],
                                     [0.1343, 0.5976],
                                     [0.1538, 0.6698]])
        self.agent_pos =  torch.tensor([[0.2628, 0.8743],
                                        [0.1437, 0.4477],
                                        [0.0622, 0.5993],
                                        [0.1133, 0.0316],
                                        [0.7639, 0.1956],
                                        [0.2182, 0.7669]])
        data = HeteroData({
            "agent": {
                "pos": self.agent_pos,
            },
            "object": {
                "pos": self.obj_pos,
                "x": self.obj_x,
            },
            ("agent", "observe", "object"): {
                "edge_index": self.obj_agent_edge_index,
            },
            ("agent", "communicate", "agent"): {
                "edge_index": self.agent_edge_index,
            }
        })
        self.data = Batch.from_data_list([data])
        self.x_idx = self.obj_agent_edge_index[0, :]
        self.obj_idx = self.obj_agent_edge_index[1, :]


    def error3_data(self):
        self.agent_edge_index = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 0],
                                              [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 0, 3]])
        self.obj_agent_edge_index = torch.tensor([[3],[0]])
        self.obj_x =  torch.tensor([[0.7246, 0.6576, 0.4876, 0.5964]])
        self.obj_pos = torch.tensor([[0.2429, 0.0285]])
        self.agent_pos =  torch.tensor([[0.5355, 0.3244],
                                        [0.7988, 0.2628],
                                        [0.6903, 0.2644],
                                        [0.1332, 0.1259]])
        data = HeteroData({
            "agent": {
                "pos": self.agent_pos,
            },
            "object": {
                "pos": self.obj_pos,
                "x": self.obj_x,
            },
            ("agent", "observe", "object"): {
                "edge_index": self.obj_agent_edge_index,
            },
            ("agent", "communicate", "agent"): {
                "edge_index": self.agent_edge_index,
            }
        })
        self.data = Batch.from_data_list([data])
        self.x_idx = self.obj_agent_edge_index[0, :]
        self.obj_idx = self.obj_agent_edge_index[1, :]


    def error4_data(self):
        self.agent_edge_index = torch.tensor([[0, 2, 1, 0, 2, 3, 1, 3, 0, 1, 2, 1, 3, 0],
                                                [0, 0, 1, 2, 2, 3, 3, 1, 1, 0, 1, 2, 0, 3]])
        self.obj_agent_edge_index = torch.tensor([[3],[0]])
        self.obj_x = torch.tensor([[0.5617, 0.2381, 0.6571, 0.8911]])
        self.obj_pos = torch.tensor([[0.6282, 0.7184]])
        self.agent_pos = torch.tensor([[0.2162, 0.1893],
                                        [0.8944, 0.4285],
                                        [0.2105, 0.0019],
                                        [0.8110, 0.9245]])
        data = HeteroData({
            "agent": {
                "pos": self.agent_pos,
            },
            "object": {
                "pos": self.obj_pos,
                "x": self.obj_x,
            },
            ("agent", "observe", "object"): {
                "edge_index": self.obj_agent_edge_index,
            },
            ("agent", "communicate", "agent"): {
                "edge_index": self.agent_edge_index,
            }
        })
        self.data = Batch.from_data_list([data])
        self.x_idx = self.obj_agent_edge_index[0, :]
        self.obj_idx = self.obj_agent_edge_index[1, :]


    def gen_all_edges(self, data):
        data_list = data.to_data_list()
        num_agent_per_batch = torch.tensor([data_list[i]['agent'].pos.shape[0] for i in range(len(data_list))])
        num_obj_per_batch = torch.tensor([data_list[i]['object'].pos.shape[0] for i in range(len(data_list))])
        cum_agent_per_batch = torch.cat([torch.zeros(1), torch.cumsum(num_agent_per_batch, dim=0)])
        cum_obj_per_batch = torch.cat([torch.zeros(1), torch.cumsum(num_obj_per_batch, dim=0)])
        edge_idx = torch.cat([
                torch.cartesian_prod(torch.arange(num_agent_per_batch[i]), torch.arange(num_obj_per_batch[i])).T +
                torch.tensor([cum_agent_per_batch[i], cum_obj_per_batch[i]])[:,None]
            for i in range(len(data_list))], dim=1).long()
        return edge_idx


    def test_agent_to_edge(self):
        encode_gnn = EncodeGNN(in_channels=self.feat_dim+2, out_channels=self.hidden_dim, max_obj=self.mean_obj*2, position="rel")
        merge_gnn = MergeGNN(in_channels=self.hidden_dim, out_channels=self.hidden_dim, orig_dim=self.feat_dim+2, max_obj=self.mean_obj*2, position="rel")

        import types
        def message(gnn_self, x_j, pos_j, pos_i):
            gnn_self.x_edge, gnn_self.x_idx_edge = gnn_self.input_decoder(x_j)
            return gnn_self.x_edge, gnn_self.x_idx_edge
        merge_gnn.message = types.MethodType(message, merge_gnn)

        encoded = encode_gnn(x=self.obj_x, edge_index=self.obj_agent_edge_index, posx=self.obj_pos, posa=self.agent_pos)
        out = merge_gnn(x=encoded, edge_index=self.agent_edge_index, pos=self.agent_pos)

        x_dec, x_idx_dec = merge_gnn.input_decoder(encoded)
        edge_index = merge_gnn.sort_edge_index(self.agent_edge_index)
        x_dec_edge, x_idx_dec_edge = merge_gnn.agents_to_edges(x=x_dec, x_idx=x_idx_dec, edge_index=edge_index)
        x_idx_agent = edge_index[1,merge_gnn.x_idx_edge]

        self.assertTrue(torch.allclose(merge_gnn.x_edge, x_dec_edge))
        self.assertTrue(torch.all(x_idx_agent == x_idx_dec_edge))


    def test_full_abs(self):
        model1 = FusionModel(input_dim=self.feat_dim, embedding_dim=self.hidden_dim, gnn_nlayers=1, position='abs', max_obj=self.mean_obj*2)
        x1, x_idx1, obj_idx1 = model1.forward_true(self.data)
        model2 = FusionModel(input_dim=self.feat_dim, embedding_dim=self.hidden_dim, gnn_nlayers=8, position='abs', max_obj=self.mean_obj*2)
        x2, x_idx2, obj_idx2 = model2.forward_true(self.data)
        # Test Expected Edge Index
        edge_index = self.gen_all_edges(self.data)
        perm1 = torch.argsort(obj_idx2, stable=True)
        x_idx2 = x_idx2[perm1]
        obj_idx2 = obj_idx2[perm1]
        x2 = x2[perm1,:]
        perm2 = torch.argsort(x_idx2, stable=True)
        x_idx2 = x_idx2[perm2]
        obj_idx2 = obj_idx2[perm2]
        x2 = x2[perm2,:]
        edge_index_pred = torch.stack([x_idx2, obj_idx2], dim=0)
        # encode_gnn = EncodeGNN(in_channels=self.feat_dim + 2, out_channels=self.hidden_dim, position="rel")
        # x_true = encode_gnn.message(self.obj_x[self.obj_idx, :], self.obj_pos[self.obj_idx, :],self.agent_pos[self.x_idx, :])
        try:
            self.assertTrue(torch.all(edge_index == edge_index_pred))
        except:
            did_element_disappear(edge_index[0,:], edge_index[1,:], x_idx2, obj_idx2, self.agent_edge_index)
            from webweb import Web
            Web(self.agent_edge_index.T.tolist()).show()
            breakpoint()


def did_element_disappear(x_idx, obj_idx, x_idx_new, obj_idx_new, edge_index):
    idx_set = {(x_idx_new[i].item(), obj_idx_new[i].item()) for i in range(x_idx_new.shape[0])}
    for i in range(x_idx.shape[0]):
        element = (x_idx[i].item(), obj_idx[i].item())
        if element not in idx_set:
            print(element)


# TODO: there is an issue with position='rel'

# Tests:
# only self loops -> same map
# integration test
# check that true forward is the same as forward