import logging

def network_builder(net_name, feature, ego_only = False):
    logger = logging.getLogger()
    if net_name == 'Recurrence_1':
        from model.recurrence_1 import flow_ae, bbox_ae, ego_ae
        logger.info(f'Build Model\t::\t{net_name}')
        return bbox_ae(feature), flow_ae(feature), ego_ae(feature)
    
    if net_name == 'Recurrence_1_SAD':
        from model.recurrence_1 import flow_SAD, bbox_SAD, ego_SAD
        logger.info(f'Build Model\t::\t{net_name}')
        return bbox_SAD(feature), flow_SAD(feature), ego_SAD(feature), ego_SAD(feature)
    
    if net_name == 'Recurrence_2':
        from model.recurrence_2 import flow_ae, bbox_ae, ego_ae
        logger.info(f'Build Model\t:\t{net_name}')
        return bbox_ae(feature), flow_ae(feature), ego_ae(feature)
    
    if net_name == 'Recurrence_2_SAD':
        from model.recurrence_2 import flow_SAD, bbox_SAD, ego_SAD
        logger.info(f'Build Model\t:\t{net_name}')
        return bbox_SAD(feature), flow_SAD(feature), ego_SAD(feature), ego_SAD(feature)