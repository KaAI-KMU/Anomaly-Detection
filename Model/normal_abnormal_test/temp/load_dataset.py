from torch.utils import data
from config.data_config import data_shuffle, num_workers, split_ratio

def dataset_loader(net_name, state = 'Pretrain', purpose = 'Train'):

    # 사전학습(Auto Encoder)
    if net_name.split('_')[0] == 'Recurrence' and state == 'Pretrain':
        from utils.Dataset_pickled_recurrence import Recurrence_Pretrain_DATASET
        train = Recurrence_Pretrain_DATASET()
        train, val = data.random_split(train, split_ratio)

        train = data.DataLoader(train, batch_size=10, shuffle=True, num_workers=num_workers)
        val = data.DataLoader(val, batch_size=10, shuffle=True, num_workers=num_workers)

        return train, val
    
    # Initilizing
    if purpose == 'Init':
        from temp.Dataset_pickled_recurrence import Recurrence_Pretrain_DATASET
        train = Recurrence_Pretrain_DATASET()
        train = data.DataLoader(train, batch_size=10, shuffle = data_shuffle, num_workers=num_workers)
        return train 

    # Recurrence Model + SAD
    elif net_name.split('_')[0] == 'Recurrence' and state == 'SAD':
        from temp.Dataset_nor_abnor_recurrence import Recurrence_SAD_DATASET_NORMAL, Recurrence_SAD_DATASET_ABNORMAL
        train_normal = Recurrence_SAD_DATASET_NORMAL()
        train_abnormal = Recurrence_SAD_DATASET_ABNORMAL()

    # Recurrence Model + SAD + Ego Only
    elif net_name.split('_')[0] == 'Recurrence' and state == 'SAD_EGO':
        from temp.Dataset_nor_abnor_recurrence import Recurrence_SAD_EGO_DATASET_NORMAL, Recurrence_SAD_EGO_DATASET_ABNORMAL
        train_normal = Recurrence_SAD_EGO_DATASET_NORMAL()
        train_abnormal = Recurrence_SAD_EGO_DATASET_ABNORMAL()
    
    if purpose == 'Train':
        train_normal, val_normal = data.random_split(train_normal, split_ratio)
        train_abnormal, val_abnormal = data.random_split(train_abnormal, split_ratio)

        train_normal = data.DataLoader(train_normal, batch_size=10//2, shuffle = data_shuffle, num_workers=num_workers)
        val_normal = data.DataLoader(val_normal, batch_size=10//2, shuffle = data_shuffle, num_workers=num_workers)
        
        train_abnormal = data.DataLoader(train_abnormal, batch_size=10//2, shuffle = data_shuffle, num_workers=num_workers)
        val_abnormal = data.DataLoader(val_abnormal, batch_size=10//2, shuffle = data_shuffle, num_workers=num_workers)
        
        return train_normal, train_abnormal, val_normal, val_abnormal
