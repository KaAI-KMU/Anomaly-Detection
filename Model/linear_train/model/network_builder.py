import logging


def network_builder(net_name, feature, ego_only = False):
    logger = logging.getLogger()
    if net_name == 'linear_1':
        from model.linear_1 import ALL_MODEL
        logger.info(f'Build Model\t::\t{net_name}')
        return ALL_MODEL(feature)
    elif net_name == 'linear_1_SAD':
        from model.linear_1 import ALL_MODEL_SAD
        logger.info(f'Build Model\t::\t{net_name}')
        return ALL_MODEL_SAD(feature)
    elif net_name == 'conv_1':
        from model.conv_1 import CONV_MODEL
        logger.info(f'Build Model\t::\t{net_name}')
        return CONV_MODEL(feature)
    elif net_name == 'conv_1_SAD':
        from model.conv_1 import CONV_MODEL_SAD
        logger.info(f'Build Model\t::\t{net_name}')
        return CONV_MODEL_SAD(feature)
    