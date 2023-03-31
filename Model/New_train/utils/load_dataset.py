from torch.utils import data
from config.data_config import data_shuffle, data_batch, num_workers

def dataset_loader(net_name, state = 'Pretrain'):
    if net_name.split('_')[0] == 'Recurrence' and state == 'Pretrain':
        from datasets.Dataset_pickled_recurrence import Recurrence_Pretrain_DATASET
        return data.DataLoader(Recurrence_Pretrain_DATASET(), batch_size=data_batch, shuffle = data_shuffle, num_workers=num_workers)

    elif net_name.split('_')[0] == 'Recurrence' and state == 'SAD':
        from datasets.Dataset_pickled_recurrence import Recurrence_SAD_DATASET
        return data.DataLoader(Recurrence_SAD_DATASET(), batch_size=data_batch, shuffle = data_shuffle, num_workers=num_workers)
    
    elif net_name.split('_')[0] == 'Recurrence' and state == 'SAD_EGO':
        from datasets.Dataset_pickled_recurrence import Recurrence_SAD_EGO_DATASET
        return data.DataLoader(Recurrence_SAD_EGO_DATASET(), batch_size=data_batch, shuffle = data_shuffle, num_workers=num_workers)
    