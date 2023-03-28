import torch
from model import network_builder
import logging

def Load_model(path, net_name): # path = result/{time}/
    SAD_other, SAD_ego = network_builder(f'{net_name}_SAD')
    logger = logging.getLogger()
    
    other_dict = torch.load(f'{path}/other.pt')
    SAD_other.load_state_dict(other_dict['net_dict'])
    SAD_other['c'] = other_dict['c']

    ego_dict = torch.load(f'{path}/ego.pt')
    SAD_ego.load_state_dict(ego_dict['net_dict'])
    SAD_ego['c'] = ego_dict['c']

    logger.info(f'Load Model weight ::\t{path}')
    
    return SAD_other, SAD_ego
