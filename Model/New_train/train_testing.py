import torch
import logging
import random
import os
from config.main_config import RESULT_PATH, pretrain_weight_path
from config.train_config import pretrain_optimzier, pretrain_lr, pretrain_weight_decay, pretrain_milestone, pretrain_gamma, pretrain_criterion, pretrain_epoch, train_optimizer, train_lr, train_weight_decay, train_milestone, train_gamma, train_epoch, eta
from datetime import datetime
from utils.load_model import Load_model
from utils.trainer import AETrain, SADTrain, EGOTrain
from utils.init_model import init_weight

pretrain_optimzier = 'Adam'
pretrain_lr = 0.001
pretrain_weight_decay = 1e-6
pretrain_milestone = [50]
pretrain_gamma = 0.1
pretrain_criterion = 'MSE'

pretrain_epoch = 100

# SAD Train Section


def main(network_name, CALLBACK):
    
    net_name = network_name
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

    logger.info(f'Network Name :: {net_name}')
    logger.info(f'Pretrain learning rate :: {pretrain_lr}')
    logger.info(f'Pretrain Optimizer :: {pretrain_optimzier}')
    logger.info(f'Pretrain weight decay :: {pretrain_weight_decay}')
    logger.info(f'Pretrain milestone :: {pretrain_milestone}')
    logger.info(f'Pretrain gamma :: {pretrain_gamma}')
    logger.info(f'Pretrain criterion :: {pretrain_criterion}')
    logger.info(f'Pretrain epoch :: {pretrain_epoch}')

    logger.info(f'Train optimizer :: {train_optimizer}')
    logger.info(f'Train learning rate :: {train_lr}')
    logger.info(f'Train weight decay :: {train_weight_decay}')
    logger.info(f'Train milestone :: {train_milestone}')
    logger.info(f'Train gamma :: {train_gamma}')
    logger.info(f'Train epoch :: {train_epoch}')
    logger.info(f'Eta :: {eta}')
    logger.info(f'Callback :: {CALLBACK}')
    
    # 모델들은 CPU에 위치
    if pretrain_weight_path:
        # 모델을 생성하고 가중치를 업로드 한 뒤 모델을 반환
        flow_model, bbox_model, ego_model, ego_model_ego_train = Load_model(pretrain_weight_path, net_name)
    else:
        # AE 모델을 생성하고 학습을 한 뒤 모델을 반환
        flow_model, bbox_model, ego_model, ego_model_ego_train = AETrain(net_name, result_path)
        other_model, ego_model = init_weight(flow_model, bbox_model, ego_model, ego_model_ego_train, net_name)
    
    net_name = f'{net_name}_SAD'

    other_model = SADTrain(other_model, net_name)
    ego_model = EGOTrain(ego_model, net_name)

    # 저장하기
    torch.save(other_model, f'{result_path}other_model.pt')
    torch.save(ego_model, f'{result_path}ego_model.pt')
    logger.info(f'Train Weight Saved\t::\t{result_path}')

    
if __name__ == '__main__':
    main()