import logging
from model.recurrence_dummy_model import dummy_flow, dummy_bbox, dummy_ego
from model.Recurrence_1 import SAD_other, SAD_ego

def network_builder(net_name, ego_only = False):
    logger = logging.getLogger()
    if net_name == 'Recurrence_1':
        logger.info(f'Build Model ::\t{net_name}')
        return dummy_bbox(), dummy_flow(), dummy_ego()
    if net_name == 'Recurrence_1_SAD':
        logger.info(f'Build Model ::\t{net_name}')
        return SAD_other(), SAD_ego()