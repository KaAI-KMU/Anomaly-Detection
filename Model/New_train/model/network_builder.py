import logging
from model.old_seperate_autoencoder_model import ego_model, flow_model, bbox_model

def network_builder(net_name, ego_only = False):
    logger = logging.getLogger()
    if net_name == 'Recurrence_1':
        logger.info(f'Build Model ::\t{net_name}')
        return bbox_model(), flow_model(), ego_model()