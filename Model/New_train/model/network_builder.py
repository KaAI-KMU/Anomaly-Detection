import logging

def network_builder(net_name, ego_only = False):
    logger = logging.getLogger()
    if net_name == 'Recurrence_1':
        logger.info(f'Build Model ::\t{net_name}')
        # 모델 3가지 호출
        # if ego_only: return ego
        pass