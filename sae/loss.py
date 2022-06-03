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





