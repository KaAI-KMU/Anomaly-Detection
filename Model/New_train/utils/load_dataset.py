import torch
from torch.utils import data

def dataset_loader(net_name):
    if net_name.split('_')[0] == 'Recurrence':
        from datasets.Dataset_recurrence import Recurrence_Pretrain_DATASET
        return data.DataLoader(Recurrence_Pretrain_DATASET())