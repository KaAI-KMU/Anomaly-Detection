import torch
from model.network_builder import network_builder
from utils.load_dataset import dataset_loader
from tqdm import tqdm
from config.model_config import feature_space
import logging

def init_weight(model, net_name, feature):
    net_name = f'{net_name}_SAD'

    logger = logging.getLogger()
    logger.info(f'Start Initializing SAD Model ::\t{net_name}')

    # SAD 모델 생성
    SAD_model = network_builder(net_name=net_name, feature = feature)

    # SAD 모델 weight
    SAD_model_dict = SAD_model.state_dict()

    # Pretrained AutoEncoder weight
    model_dict = model.state_dict()

    # AutoEncoder weight중 겹치는 weight만 떼어내기
    model_dict = {k: v for k, v in model_dict.items() if k in SAD_model_dict}

    # Default Task를 위한 모델 weight 업로드
    SAD_model.load_state_dict(model_dict)

    model_c = init_center_c(SAD_model, net_name, feature)

    SAD_model.c = (model_c['c'])

    logger.info(f'Initializing Done')
    return SAD_model

def init_center_c(SAD_other, net_name,feature, eps = 0.1):

    logger = logging.getLogger()
    logger.info(f'Start Initializing C point ::\t{net_name}')

    train_generator = dataset_loader(net_name, purpose = 'Init')
    length = len(train_generator)
    loader = tqdm(train_generator, total = length)

    logger.info(f'Initializing Data Length ::\t{length}')

    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'
    for _, param in SAD_other.named_parameters():
        pass
    c_other = torch.zeros(param.shape[0], device=device)
    n_samples = 0

    SAD_other.eval()
    SAD_other.to(device)

    with torch.no_grad():
        for data in loader:
            other_output = SAD_other(data)
            n_samples += data.shape[0]

            c_other += torch.sum(other_output, dim = 0).squeeze()

    c_other /= n_samples
    
    #c_other[(abs(c_other) < eps) & (c_other < 0)] = -eps
    #c_ego[(abs(c_ego) < eps) & (c_ego < 0)] = -eps

    c_other = {'c' : c_other}
    
    return c_other