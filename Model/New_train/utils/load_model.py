import torch
import logging

def Load_model(path): # path = result/{time}/

    logger = logging.getLogger()
    SAD_other = torch.load(f'{path}/other_model.pt')
    SAD_ego = torch.load(f'{path}/ego_model.pt')
    logger.info(f'Load Model weight ::\t{path}')
    
    return SAD_other, SAD_ego
