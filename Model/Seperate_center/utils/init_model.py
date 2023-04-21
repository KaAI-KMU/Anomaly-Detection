import torch
from model.network_builder import network_builder
from utils.load_dataset import dataset_loader
from tqdm import tqdm
from config.model_config import feature_space
import logging

def init_weight(flow_model, bbox_model, ego_model, ego_model_ego_train, net_name, feature):
    net_name = f'{net_name}_SAD'

    logger = logging.getLogger()
    logger.info(f'Start Initializing SAD Model ::\t{net_name}')

    # SAD 모델 생성
    bbox_SAD, flow_SAD, ego_SAD, ego_only_SAD = network_builder(net_name=net_name, feature = feature)

    # SAD 모델 weight
    bbox_SAD_dict = bbox_SAD.state_dict()
    flow_SAD_dict = flow_SAD.state_dict()
    ego_SAD_dict = ego_SAD.state_dict()
    ego_only_SAD_dict = ego_only_SAD.state()

    # Pretrained AutoEncoder weight
    flow_model_dict = flow_model.state_dict()
    bbox_model_dict = bbox_model.state_dict()
    ego_model_dict = ego_model.state_dict()

    # Ego Task를 위한 AutoEncoder weight
    ego_model_ego_train_dict = ego_model_ego_train.state_dict()

    # AutoEncoder weight중 겹치는 weight만 떼어내기
    flow_model_dict = {k: v for k, v in flow_model_dict.items() if k in flow_SAD_dict}
    bbox_model_dict = {k: v for k, v in bbox_model_dict.items() if k in bbox_SAD_dict}
    ego_model_dict = {k: v for k, v in ego_model_dict.items() if k in ego_SAD_dict}
    # 하나로 합치는 과정

    # Ego Task를 위한 model의 겹치는 weight만 떼어내기
    ego_model_ego_train_dict = {k: v for k, v in ego_model_ego_train_dict.items() if k in ego_only_SAD_dict}

    # Default Task를 위한 모델 weight 업로드
    bbox_SAD.load_state_dict(bbox_model_dict)
    flow_SAD.load_state_dict(flow_SAD_dict)
    ego_SAD.load_state_dict(ego_SAD_dict)
    # Ego Task를 위한 모델 weight 업로드
    ego_only_SAD.load_state_dict(ego_model_ego_train_dict)

    c_bbox, c_flow, c_ego, c_ego_only = init_center_c(bbox_SAD, flow_SAD, ego_SAD, ego_only_SAD, net_name, feature)

    bbox_SAD.c = c_bbox
    flow_SAD.c = c_flow
    ego_SAD.c = c_ego
    ego_only_SAD.c = c_ego_only
    logger.info(f'Initializing Done')
    return bbox_SAD, flow_SAD, ego_SAD, ego_only_SAD

def init_center_c(bbox_SAD, flow_SAD, ego_SAD, ego_only_SAD, net_name, feature, eps = 0.1):

    logger = logging.getLogger()
    logger.info(f'Start Initializing C point ::\t{net_name}')

    train_generator = dataset_loader(net_name, purpose = 'Init')
    length = len(train_generator)
    loader = tqdm(train_generator, total = length)

    logger.info(f'Initializing Data Length ::\t{length}')

    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'

    c_bbox = torch.zeros(feature, device=device)
    c_flow = torch.zeros(feature, device=device)
    c_ego = torch.zeros(feature, device=device)
    c_ego_only = torch.zeros(feature, device=device)
    
    n_samples = 0

    bbox_SAD.eval()
    flow_SAD.eval()
    ego_SAD.eval()
    ego_only_SAD.eval()
    bbox_SAD.to(device)
    flow_SAD.to(device)
    ego_SAD.to(device)
    ego_only_SAD.to(device)

    with torch.no_grad():
        for data in loader:
            bbox, flow, ego = data

            bbox_output = bbox_SAD(bbox)
            flow_output = flow_SAD(flow)
            ego_output = ego_SAD(ego)
            ego_only_output = ego_only_SAD(ego)
            

            n_samples += bbox.shape[0]
            c_bbox += torch.sum(bbox_output, dim = 0).squeeze()
            c_flow += torch.sum(flow_output, dim = 0).squeeze()
            c_ego += torch.sum(ego_output, dim = 0).squeeze()
            c_ego_only += torch.sum(ego_only_output, dim = 0).squeeze()

    c_bbox /= n_samples
    c_flow /= n_samples
    c_ego /= n_samples
    c_ego_only /= n_samples
    

    c_bbox[(abs(c_bbox) < eps) & (c_bbox < 0)] = -eps
    c_flow[(abs(c_flow) < eps) & (c_flow < 0)] = -eps
    c_ego[(abs(c_ego) < eps) & (c_ego < 0)] = -eps
    c_ego_only[(abs(c_ego_only) < eps) & (c_ego_only < 0)] = -eps

    
    return c_bbox, c_flow, c_ego, c_ego_only