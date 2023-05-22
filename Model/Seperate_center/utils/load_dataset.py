from torch.utils import data
from config.data_config import data_shuffle, data_batch, num_workers, split_ratio

def dataset_loader(net_name, state = 'Pretrain', purpose = 'Train'):
    if net_name.split('_')[0] == 'Recurrence' and state == 'Pretrain':
        from datasets.Dataset_pickled_recurrence import Recurrence_Pretrain_DATASET
        train = Recurrence_Pretrain_DATASET()

    elif net_name.split('_')[0] == 'Recurrence' and state == 'SAD':
        from datasets.Dataset_pickled_recurrence import Recurrence_SAD_DATASET
        train = Recurrence_SAD_DATASET()

    elif net_name.split('_')[0] == 'Recurrence' and state == 'SAD_EGO':
        from datasets.Dataset_pickled_recurrence import Recurrence_SAD_EGO_DATASET
        train = Recurrence_SAD_EGO_DATASET()
    
    if purpose == 'Train':
        train, val = data.random_split(train, split_ratio)
        train = data.DataLoader(train, batch_size=data_batch, shuffle = data_shuffle, num_workers=num_workers)
        val = data.DataLoader(val, batch_size=data_batch, shuffle = data_shuffle, num_workers=num_workers)
        return train, val
    elif purpose == 'Init':
        train = data.DataLoader(train, batch_size=data_batch, shuffle = data_shuffle, num_workers=num_workers)
        return train 
