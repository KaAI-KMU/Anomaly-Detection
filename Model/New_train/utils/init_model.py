import torch
from model.network_builder import network_builder
from utils.load_dataset import dataset_loader
from tqdm import tqdm
from config.model_config import feature_space
import logging

def init_weight(flow_model, bbox_model, ego_model, ego_model_ego_train, net_name):
    net_name = f'{net_name}_SAD'

    logger = logging.getLogger()
    logger.info(f'Start Initializing SAD Model ::\t{net_name}')

    # SAD 모델 생성
    SAD_other, SAD_ego = network_builder(net_name=net_name)

    # SAD 모델 weight
    SAD_other_dict = SAD_other.state_dict()
    SAD_ego_dict = SAD_ego.state_dict()

    # Pretrained AutoEncoder weight
    flow_model_dict = flow_model.state_dict()
    bbox_model_dict = bbox_model.state_dict()
    ego_model_dict = ego_model.state_dict()

    # Ego Task를 위한 AutoEncoder weight
    ego_model_ego_train_dict = ego_model_ego_train.state_dict()

    # AutoEncoder weight중 겹치는 weight만 떼어내기
    flow_model_dict = {k: v for k, v in flow_model_dict.items() if k in SAD_other_dict}
    bbox_model_dict = {k: v for k, v in bbox_model_dict.items() if k in SAD_other_dict}
    ego_model_dict = {k: v for k, v in ego_model_dict.items() if k in SAD_other_dict}
    # 하나로 합치는 과정
    flow_model_dict.update(bbox_model_dict)
    flow_model_dict.update(ego_model_dict)

    # Ego Task를 위한 model의 겹치는 weight만 떼어내기
    ego_model_ego_train_dict = {k: v for k, v in ego_model_ego_train_dict.items() if k in SAD_ego_dict}

    # Default Task를 위한 모델 weight 업로드
    SAD_other.load_state_dict(flow_model_dict)
    # Ego Task를 위한 모델 weight 업로드
    SAD_ego.load_state_dict(ego_model_ego_train_dict)

    SAD_other_c, SAD_ego_c = init_center_c(SAD_other, SAD_ego, net_name)

    SAD_other.c = (SAD_other_c['c'])
    SAD_ego.c = (SAD_ego_c['c'])
    logger.info(f'Initializing Done')
    return SAD_other, SAD_ego

def init_center_c(SAD_other, SAD_ego, net_name, eps = 0.1):

    logger = logging.getLogger()
    logger.info(f'Start Initializing C point ::\t{net_name}')

    train_generator = dataset_loader(net_name)
    length = len(train_generator)
    loader = tqdm(train_generator, total = length)

    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'

    c_other = torch.zeros(feature_space, device=device)
    c_ego = torch.zeros(feature_space, device=device)
    n_samples = 0

    SAD_other.eval()
    SAD_ego.eval()
    SAD_other.to(device)
    SAD_ego.to(device)

    with torch.no_grad():
        for data in loader:
            bbox, flow, ego = data

            other_output = SAD_other(bbox, flow, ego)
            ego_output = SAD_ego(ego)

            n_samples += bbox.shape[0]
            c_other += torch.sum(other_output, dim = 0).squeeze()
            c_ego += torch.sum(ego_output, dim = 0).squeeze()

    c_other /= n_samples
    c_ego /= n_samples

    c_other[(abs(c_other) < eps) & (c_other < 0)] = -eps
    c_ego[(abs(c_ego) < eps) & (c_ego < 0)] = -eps

    c_other = {'c' : c_other}
    c_ego = {'c' : c_ego}
    
    return c_other, c_ego