import unittest
from fusion_dataset import ObsEnv
from fusion_model import FusionModel
from fusion_gnn import MergeGNN, EncodeGNN
import torch
from torch_geometric.data import HeteroData, Batch, Data
from torch_scatter import scatter
from sae import batch_to_set_lens

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


    def random_data(self):
        env = ObsEnv(mean_agents=self.mean_agents, mean_objects=self.mean_obj, feat_dim=self.feat_dim)
        data = env.sample_n(10)
        return data


    def error1_data(self):
        agent_edge_index = torch.tensor([[0, 5, 1, 2, 1, 2, 5, 3, 4, 0, 2, 5, 1, 5, 0, 2, 3, 1, 1, 0, 2, 3, 1, 4],
                                         [0, 0, 1, 1, 2, 2, 2, 3, 4, 5, 5, 5, 5, 1, 2, 0, 1, 3, 0, 1, 3, 2, 4, 1]])
        obj_agent_edge_index = torch.tensor([[0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 5, 5, 5, 5],
                                             [0, 1, 4, 3, 4, 0, 1, 3, 4, 2, 0, 1, 3, 4]])
        obj_x = torch.tensor([[0.1739, 0.1655, 0.3468, 0.2432],
                              [0.6342, 0.5083, 0.7763, 0.3976],
                              [0.1338, 0.6652, 0.8248, 0.8279],
                              [0.9121, 0.2734, 0.4609, 0.6362],
                              [0.4090, 0.6820, 0.6805, 0.4833]])
        obj_pos = torch.tensor([[0.1184, 0.7796],
                                [0.0272, 0.7945],
                                [0.0677, 0.0017],
                                [0.1343, 0.5976],
                                [0.1538, 0.6698]])
        agent_pos = torch.tensor([[0.2628, 0.8743],
                                  [0.1437, 0.4477],
                                  [0.0622, 0.5993],
                                  [0.1133, 0.0316],
                                  [0.7639, 0.1956],
                                  [0.2182, 0.7669]])
        data = HeteroData({
            "agent": {
                "pos": agent_pos,
            },
            "object": {
                "pos": obj_pos,
                "x": obj_x,
            },
            ("agent", "observe", "object"): {
                "edge_index": obj_agent_edge_index,
            },
            ("agent", "communicate", "agent"): {
                "edge_index": agent_edge_index,
            }
        })
        data = Batch.from_data_list([data])
        return data


    def error2_data(self):
        agent_edge_index = torch.tensor([[0, 2, 1, 0, 2, 3, 1, 3, 0, 1, 2, 1, 3, 0],
                                         [0, 0, 1, 2, 2, 3, 3, 1, 1, 0, 1, 2, 0, 3]])
        obj_agent_edge_index = torch.tensor([[3],[0]])
        obj_x = torch.tensor([[0.5617, 0.2381, 0.6571, 0.8911]])
        obj_pos = torch.tensor([[0.6282, 0.7184]])
        agent_pos = torch.tensor([[0.2162, 0.1893],
                                  [0.8944, 0.4285],
                                  [0.2105, 0.0019],
                                  [0.8110, 0.9245]])
        data = HeteroData({
            "agent": {
                "pos": agent_pos,
            },
            "object": {
                "pos": obj_pos,
                "x": obj_x,
            },
            ("agent", "observe", "object"): {
                "edge_index": obj_agent_edge_index,
            },
            ("agent", "communicate", "agent"): {
                "edge_index": agent_edge_index,
            }
        })
        data = Batch.from_data_list([data])
        return data


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


    def run_agent_to_edge(self, data, position="abs"):
        obj_x = data['object'].x
        obj_pos = data['object'].pos
        agent_pos = data['agent'].pos
        obj_agent_edge_index = data[("agent", "observe", "object")].edge_index
        agent_edge_index = data[("agent", "communicate", "agent")].edge_index

        encode_gnn = EncodeGNN(in_channels=self.feat_dim+2, out_channels=self.hidden_dim, max_obj=self.mean_obj*2, position=position)
        merge_gnn = MergeGNN(in_channels=self.hidden_dim, out_channels=self.hidden_dim, orig_dim=self.feat_dim+2, max_obj=self.mean_obj*2, position=position)

        import types
        def message(gnn_self, x_i, pos_i, pos_j, idx_i, idx_j):
            gnn_self.x_edge, gnn_self.x_idx_edge = gnn_self.input_decoder(x_i) # i is the src, j is self

            gnn_self.src_idx = gnn_self.edge_index[1, gnn_self.x_idx_edge]  # = idx_i.repeat_interleave(objs_per_agent)
            gnn_self.agent_idx = gnn_self.edge_index[0, gnn_self.x_idx_edge]  # = idx_j.repeat_interleave(objs_per_agent)
            ope = scatter(src=torch.ones(gnn_self.x_idx_edge.shape[0]), index=gnn_self.x_idx_edge,dim_size=idx_i.shape[0]).long()
            src_idx_exp = idx_i.repeat_interleave(ope)
            agent_idx_exp = idx_j.repeat_interleave(ope)
            edge_data = gnn_self.agents_to_edges(gnn_self.values["x_pred"][-1], gnn_self.values["batch_pred"][-1],gnn_self.edge_index,obj_idx=torch.arange(gnn_self.values["x_pred"][-1].shape[0]))
            xe = edge_data["x"]
            ae_idx = edge_data["agent_idx"]
            srce_idx = edge_data["agent_src_idx"]
            pose_i = gnn_self.pos[srce_idx, :]
            pose_j = gnn_self.pos[ae_idx, :]
            self.assertTrue(torch.all(gnn_self.src_idx == srce_idx))
            self.assertTrue(torch.all(src_idx_exp == srce_idx))
            self.assertTrue(torch.all(gnn_self.agent_idx == ae_idx))
            self.assertTrue((torch.all(agent_idx_exp == ae_idx)))
            self.assertTrue(torch.allclose(gnn_self.x_edge, xe))

            if gnn_self.position == 'rel':
                ope = scatter(src=torch.ones(gnn_self.x_idx_edge.shape[0]), index=gnn_self.x_idx_edge,dim_size=idx_i.shape[0]).long()
                pos_i_exp = pos_i.repeat_interleave(ope, dim=0)
                pos_j_exp = pos_j.repeat_interleave(ope, dim=0)
                gnn_self.x_edge = gnn_self.update_rel_pos(gnn_self.x_edge, pos_i_exp, pos_j_exp)
                self.assertTrue(torch.all(pose_i == pos_i_exp))
                self.assertTrue(torch.all(pose_j == pos_j_exp))

            # orig_out1, orig_out2 = gnn_self.message_orig(x_i=x_i, pos_i=pos_i, pos_j=pos_j, idx_i=idx_i, idx_j=idx_j)
            # self.assertTrue(torch.allclose(orig_out1, gnn_self.x_edge))
            # self.assertTrue(torch.all(orig_out2 == gnn_self.x_idx_edge))
            return gnn_self.x_edge, gnn_self.x_idx_edge

        # merge_gnn.message_orig = types.MethodType(merge_gnn.message, merge_gnn)
        merge_gnn.message = types.MethodType(message, merge_gnn)

        encoded = encode_gnn(x=obj_x, edge_index=obj_agent_edge_index, posx=obj_pos, posa=agent_pos)
        out = merge_gnn(x=encoded, edge_index=agent_edge_index, pos=agent_pos)

        x_dec, x_idx_dec = merge_gnn.input_decoder(encoded)
        edge_index = merge_gnn.sort_edge_index(agent_edge_index)
        edge_data = merge_gnn.agents_to_edges(x=x_dec, agent_idx=x_idx_dec, edge_index=edge_index)
        x_dec_edge = edge_data["x"]
        agent_idx_dec_edge = edge_data["agent_idx"]
        src_idx_dec_edge = edge_data["agent_src_idx"]

        merge_gnn_agent_idx = edge_index[0,merge_gnn.x_idx_edge]
        if merge_gnn.position == "rel":
            x_dec_edge[:,-2:] = x_dec_edge[:,-2:] + agent_pos[src_idx_dec_edge,:] - agent_pos[agent_idx_dec_edge,:]

        self.assertTrue(torch.allclose(merge_gnn.x_edge, x_dec_edge))
        self.assertTrue(torch.all(merge_gnn_agent_idx == agent_idx_dec_edge))


    def run_full_truth(self, data, position="abs"):
        model = FusionModel(input_dim=self.feat_dim, embedding_dim=self.hidden_dim, gnn_nlayers=8, position=position, max_obj=self.mean_obj*2)
        x, agent_idx, obj_idx = model.forward_true(data)
        perm1 = torch.argsort(obj_idx, stable=True)
        agent_idx = agent_idx[perm1]
        obj_idx = obj_idx[perm1]
        perm2 = torch.argsort(agent_idx, stable=True)
        agent_idx = agent_idx[perm2]
        obj_idx = obj_idx[perm2]
        edge_index_pred = torch.stack([agent_idx, obj_idx], dim=0)
        x = x[perm1, :]
        x = x[perm2, :]

        edge_index = self.gen_all_edges(data)
        obj_x_true = data['object'].x
        obj_pos_true = data['object'].pos
        agent_pos_true = data['agent'].pos
        x_true = model.encode_gnn.message(obj_x_true[edge_index[1,:], :], obj_pos_true[edge_index[1,:], :], agent_pos_true[edge_index[0,:], :])
        self.assertTrue(torch.all(edge_index == edge_index_pred))
        self.assertTrue(torch.allclose(x, x_true))


    def test_full_truth(self):
        error1_data = self.error1_data()
        self.run_full_truth(error1_data, position="abs")
        self.run_full_truth(error1_data, position="rel")
        error2_data = self.error2_data()
        self.run_full_truth(error2_data, position="abs")
        self.run_full_truth(error2_data, position="rel")
        random_data = self.random_data()
        self.run_full_truth(random_data, position="abs")
        self.run_full_truth(random_data, position="rel")


    def test_agent_to_edge(self):
        random_data = self.random_data()
        self.run_agent_to_edge(random_data, position="abs")
        self.run_agent_to_edge(random_data, position="rel")
        error1_data = self.random_data()
        self.run_agent_to_edge(error1_data, position="abs")
        self.run_agent_to_edge(error1_data, position="rel")
        error2_data = self.random_data()
        self.run_agent_to_edge(error2_data, position="abs")
        self.run_agent_to_edge(error2_data, position="rel")


def did_element_disappear(x_idx, obj_idx, x_idx_new, obj_idx_new, edge_index):
    idx_set = {(x_idx_new[i].item(), obj_idx_new[i].item()) for i in range(x_idx_new.shape[0])}
    for i in range(x_idx.shape[0]):
        element = (x_idx[i].item(), obj_idx[i].item())
        if element not in idx_set:
            print(element)


# Tests:
# only self loops -> same map
# integration test
# check that true forward is the same as forward
# loss is 0