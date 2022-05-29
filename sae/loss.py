from torch.nn import MSELoss, CrossEntropyLoss
import torch
import numpy as np


def get_loss_idxs(set1_lens, set2_lens):
    """Given two data of different sizes, return the indices for the
    first and second datas denoting the elements to be used in the loss
    function. setn_lens represents the number of elements in each batch.

    E.g. 
    >>> pred_idx, target_idx = get_loss_idxs(pred, target, ....)
    >>> loss = functional.mse_loss(pred[pred_idx], target[target_idx])
    """
    assert set1_lens.shape == set2_lens.shape
    set_lens = torch.min(set1_lens, set2_lens)

    ptr1_start = set1_lens.cumsum(dim=0).roll(1)
    ptr1_start[0] = 0
    ptr1_end = ptr1_start + set_lens

    ptr2_start = set2_lens.cumsum(dim=0).roll(1)
    ptr2_start[0] = 0
    ptr2_end = ptr2_start + set_lens

    ptr1 = torch.cat(
        [
            torch.arange(ptr1_start[i], ptr1_end[i], device=set1_lens.device) 
            for i in range(ptr1_start.numel())
        ]
    )
    ptr2 = torch.cat(
        [
            torch.arange(ptr2_start[i], ptr2_end[i], device=set2_lens.device) 
            for i in range(ptr2_start.numel())
        ]
    )
    return ptr1, ptr2

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



