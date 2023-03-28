import torch
import logging
import random
import os
from config.main_config import *
from datetime import datetime
from utils.load_model import Load_model
from utils.trainer import AETrain, SADTrain, EGOTrain
from utils.init_model import init_weight

def main():
    start = datetime.now()
    start = start.strftime('%Y_%m_%d_%H_%M_%S')
    result_path = f'{RESULT_PATH}{start}/'

    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = result_path + f'log_{start}.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f'Network Name :: {network_name}')

    if not torch.cuda.is_available():
        device = 'cpu'
    
    # 모델들은 CPU에 위치
    if pretrain_weight_path:
        # 모델을 생성하고 가중치를 업로드 한 뒤 모델을 반환
        flow_model, bbox_model, ego_model, ego_model_ego_train = Load_model(pretrain_weight_path, network_name)
    else:
        # AE 모델을 생성하고 학습을 한 뒤 모델을 반환
        flow_model, bbox_model, ego_model, ego_model_ego_train = AETrain(network_name, result_path)
        other_model, ego_model = init_weight(flow_model, bbox_model, ego_model, ego_model_ego_train, network_name)
    
    network_name = f'{network_name}_SAD'

    SADTrain(other_model, network_name)
    EGOTrain(ego_model)
    
if __name__ == '__main__':
    main()