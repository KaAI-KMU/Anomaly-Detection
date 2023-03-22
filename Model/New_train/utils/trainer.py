import torch
from model import network_builder
import logging
from config.main_config import *
from utils.load_pretrain_argument import load_optimizer, load_multistep_lr, load_criterion
import time

def AETrain(net_name):
    bbox_model, flow_model, ego_model = network_builder(net_name, ego_only = False)
    logger = logging.getLogger()
    logger.info(f'Pretrain Start ::\t{net_name}')

    optimizer_bbox = load_optimizer(bbox_model.parameters())
    scheduler_bbox = load_multistep_lr(optimizer_bbox)

    optimizer_flow = load_optimizer(flow_model.parameters())
    scheduler_flow = load_multistep_lr(optimizer_flow)

    optimizer_ego = load_optimizer(ego_model.parameters())
    scheduler_ego = load_multistep_lr(optimizer_ego)

    criterion = load_criterion()

    bbox_model = bbox_model.to(device)
    flow_model = flow_model.to(device)
    ego_model = ego_model.to(device)

    criterion = criterion.to(device)

    start_time = time.time()
    for epoch in range(pretrain_epoch):
        bbox_model.train()
        flow_model.train()
        ego_model.train()

        if network_name.split('_')[0] == 'Recurrence':
            from datasets.Dataset_recurrence import Recurrence_DATASET
            


    
    

def SADTrain(flow_model, bbox_model, ego_model):
    pass

def EGOTrain(ego_model):
    pass