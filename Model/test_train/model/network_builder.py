import logging
from model.recurrence_dummy_model import dummy_flow, dummy_bbox, dummy_ego


def network_builder(net_name, feature, ego_only = False):
    logger = logging.getLogger()
    if net_name == 'Recurrence_1':
        from model.recurrence_1 import flow_ae, bbox_ae, ego_ae
        logger.info(f'Build Model\t::\t{net_name}')
        return bbox_ae(feature), flow_ae(feature), ego_ae(feature)
    
    if net_name == 'Recurrence_1_SAD':
        from model.recurrence_1 import other_SAD, ego_SAD
        logger.info(f'Build Model\t::\t{net_name}')
        return other_SAD(feature), ego_SAD(feature)
    
    if net_name == 'Recurrence_2':
        from model.recurrence_2 import flow_ae, bbox_ae, ego_ae
        logger.info(f'Build Model\t:\t{net_name}')
        return bbox_ae(feature), flow_ae(feature), ego_ae(feature)
    
    if net_name == 'Recurrence_2_SAD':
        from model.recurrence_2 import other_SAD, ego_SAD
        logger.info(f'Build Model\t:\t{net_name}')
        return other_SAD(feature), ego_SAD(feature)