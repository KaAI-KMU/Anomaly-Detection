import torch
from torch.utils import data
from config.data_config import data_shuffle, data_batch, num_workers

def dataset_loader(net_name):
    if net_name.split('_')[0] == 'Recurrence':
        from datasets.Dataset_recurrence import Recurrence_Pretrain_DATASET
        return data.DataLoader(Recurrence_Pretrain_DATASET(), batch_size=data_batch, shuffle = data_shuffle, num_workers=num_workers)