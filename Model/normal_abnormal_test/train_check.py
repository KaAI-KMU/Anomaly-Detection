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

# SAD Train Section


def main(network_name, CALLBACK, folder_name):
    
    
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
    
    
    net_name = f'{net_name}'

    bbox = torch.load(f'RESULT/{folder_name}/pretrain/bbox.pt')
    flow = torch.load(f'RESULT/{folder_name}/pretrain/flow.pt')
    ego = torch.load(f'RESULT/{folder_name}/pretrain/ego.pt')
    try:
        feature = next(bbox.bbox_decoder[1].parameters()).size()[1]
    except:
        feature = next(bbox.bbox_decoder[0].parameters()).size()[1]

    other_model, ego_model = init_weight(flow, bbox, ego, ego, net_name, feature)

    other_model = SADTrain(other_model, net_name, CALLBACK)
    ego_model = EGOTrain(ego_model, net_name, CALLBACK)

    # 저장하기
    torch.save(other_model, f'{result_path}other_model.pt')
    torch.save(ego_model, f'{result_path}ego_model.pt')
    logger.info(f'Train Weight Saved\t::\t{result_path}')

    
if __name__ == '__main__':
    model_name = ['Recurrence_1', 'Recurrence_1', 'Recurrence_1', 'Recurrence_1', 'Recurrence_2', 'Recurrence_2', 'Recurrence_2', 'Recurrence_2']
    callback = [True, True, False, False, True, True, False, False]
    folder_names = ['2023_04_10_10_11_58', 
                    '2023_04_16_13_20_43', 
                    '2023_04_15_05_51_25', 
                    '2023_04_16_17_13_12', 
                    '2023_04_16_07_08_51', 
                    '2023_04_16_21_07_37', 
                    '2023_04_15_10_52_29', 
                    '2023_04_17_02_44_46']

    if len(model_name) == len(callback) and len(callback) == len(folder_names):
        for i in range(len(model_name)):
            main(model_name[i], callback[i], folder_names[i])
    else:
        print('Fail')