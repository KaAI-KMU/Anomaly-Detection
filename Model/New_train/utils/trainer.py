import torch
from model.network_builder import network_builder
import logging
from config.main_config import *
from utils.load_pretrain_argument import load_optimizer, load_multistep_lr, load_criterion
from utils.load_dataset import dataset_loader
import time
from tqdm import tqdm
import copy

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

    train_time = time.time()
    for epoch in range(pretrain_epoch):
        bbox_model.train()
        flow_model.train()
        ego_model.train()

        logger.info(f"Start Load Training Data")
        loading_time = time.time()
        train_generator = dataset_loader(network_name)
        loading_time = time.time() - loading_time
        print(f'Loading Time :: {loading_time}')
        length = len(train_generator)
        loader = tqdm(train_generator, total = length)

        logger.info(f"Pretrain Start :: {epoch+1} epoch")
        batch_number = 0
        epoch_time = time.time()
        
        bbox_avg_loss = 0
        flow_avg_loss = 0
        ego_avg_loss = 0

        for data in loader:
            bbox_data, flow_data, ego_data = data

            # BBox Data Train
            optimizer_bbox.zero_grad()
            prediction = bbox_model(bbox_data)
            bbox_loss = criterion(prediction, bbox_data)
            bbox_loss.mean().backward()
            optimizer_bbox.step()
            bbox_avg_loss += bbox_loss.mean()/length
            # Flow Data Train
            optimizer_flow.zero_grad()
            prediction = flow_model(flow_data)
            flow_loss = criterion(prediction, flow_data)
            flow_loss.mean().backward()
            optimizer_flow.step()
            flow_avg_loss += flow_loss.mean()/length
            # Ego Data Train
            optimizer_ego.zero_grad()
            prediction = ego_model(ego_data)
            ego_loss = criterion(prediction, ego_data)
            ego_loss.mean().backward()
            optimizer_ego.step()
            ego_avg_loss += ego_loss.mean()/length

            batch_number += 1

        scheduler_bbox.step()
        scheduler_flow.step()
        scheduler_ego.step()

        epoch_time = time.time() - epoch_time
        logger.info(f'Pretrain AutoEncoder :: {epoch+1}/{pretrain_epoch} :: Train Time :: {epoch_time:.3f}s '
                    f'BBox loss :: {bbox_avg_loss} Flos loss :: {flow_avg_loss} Ego Loss :: {ego_avg_loss}')

    train_time = time.time() - train_time
    logger.info(f'Pretrain Finished :: {train_time}')
    return bbox_model, flow_model, ego_model, copy.deepcopy(ego_model)

def SADTrain(flow_model, bbox_model, ego_model):
    pass

def EGOTrain(ego_model):
    pass