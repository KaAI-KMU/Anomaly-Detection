import torch
from model import network_builder
import logging

def Load_model(path, net_name, ego_only = False): # path = result/{time}/
    bbox_model, flow_model, ego_model = network_builder(net_name, ego_only)
    logger = logging.getLogger()
    
    bbox_dict = torch.load(f'{path}/pretrain/bbox.pt')
    bbox_model.load_state_dict(bbox_dict)

    flow_dict = torch.load(f'{path}/pretrain/flow.pt')
    flow_model.load_state_dict(flow_dict)

    ego_dict = torch.load(f'{path}/pretrain/ego.pt')
    ego_model.load_state_dict(ego_dict)

    logger.info(f'Load Model weight ::\t{path}')
    
    return bbox_model, flow_model, ego_model, ego_model