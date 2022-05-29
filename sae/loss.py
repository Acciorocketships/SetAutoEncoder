from torch.nn import MSELoss, CrossEntropyLoss
import torch
import numpy as np

def mse_sparse(data1, data2):
    if data1.batch.shape[0] == data2.batch.shape[0] and torch.all(data1.batch == data2.batch):
        return MSELoss()(data1.x, data2.x)
    else:
        data1_list = data1.to_data_list()
        data2_list = data2.to_data_list()
        assert len(data1_list) == len(data2_list), "Batch sizes for the inputs must be the same. " \
                                                   "Received {data1_len} and {data2_len}"\
                                                    .format(data1_len=len(data1_list), data2_len=len(data2_list))
        batch_loss = torch.zeros(len(data1_list))
        for i in range(len(data1_list)):
            x1i = data1_list[i].x
            x2i = data2_list[i].x
            length = min(x1i.shape[0], x2i.shape[0])
            x1i_trunc = x1i[:length,:]
            x2i_trunc = x2i[:length,:]
            batch_loss[i] = MSELoss()(x1i_trunc, x2i_trunc)
            if torch.isnan(batch_loss[i]):
                batch_loss[i] = 0
        return torch.mean(batch_loss)


def corr_sparse(data1, data2):
    if data1.batch.shape[0] == data2.batch.shape[0] and torch.all(data1.batch == data2.batch):
        corr = np.corrcoef(data1.x.detach().reshape(-1), data2.x.detach().reshape(-1))[0, 1]
        return corr
    else:
        data1_list = data1.to_data_list()
        data2_list = data2.to_data_list()
        assert len(data1_list) == len(data2_list), "Batch sizes for the inputs must be the same. " \
                                                   "Received {data1_len} and {data2_len}"\
                                                    .format(data1_len=len(data1_list), data2_len=len(data2_list))
        batch_corr = torch.zeros(len(data1_list))
        for i in range(len(data1_list)):
            x1i = data1_list[i].x
            x2i = data2_list[i].x
            length = min(x1i.shape[0], x2i.shape[0])
            if length == 0:
                batch_corr[i] = 1.
            else:
                x1i_trunc = x1i[:length,:]
                x2i_trunc = x2i[:length,:]
                batch_corr[i] = np.corrcoef(x1i_trunc.detach().reshape(-1), x2i_trunc.detach().reshape(-1))[0, 1]
        return torch.mean(batch_corr)



