import torch
from model.network_builder import network_builder
import logging
from config.train_config import *
from config.main_config import device, network_name
from utils.load_train_argument import load_pretrain_optimizer, load_pretrain_multistep_lr, load_criterion, load_train_optimizer, load_train_multistep_lr
from utils.load_dataset import dataset_loader
import time
from tqdm import tqdm
import copy
import os

def AETrain(net_name, result_path):
    bbox_model, flow_model, ego_model = network_builder(net_name, ego_only = False)
    logger = logging.getLogger()
    logger.info(f'Pretrain Start\t::\t{net_name}')

    optimizer_bbox = load_pretrain_optimizer(bbox_model.parameters())
    scheduler_bbox = load_pretrain_multistep_lr(optimizer_bbox)

    optimizer_flow = load_pretrain_optimizer(flow_model.parameters())
    scheduler_flow = load_pretrain_multistep_lr(optimizer_flow)

    optimizer_ego = load_pretrain_optimizer(ego_model.parameters())
    scheduler_ego = load_pretrain_multistep_lr(optimizer_ego)

    criterion = load_criterion()

    bbox_model = bbox_model.to(device)
    flow_model = flow_model.to(device)
    ego_model = ego_model.to(device)

    criterion = criterion.to(device)

    train_time = time.time()
    train_generator, validation_generator = dataset_loader(network_name)
    length = len(train_generator)
    length_val = len(validation_generator)
    logger.info(f'Pretrain Training Data Length\t::\t{length}')
    logger.info(f'Pretrain Validation Data Length\t::\t{length_val}')
    logger.info(f'Pretrain Training Epoch\t::\t{pretrain_epoch}')
    for epoch in range(pretrain_epoch):
        bbox_model.train()
        flow_model.train()
        ego_model.train()
        criterion.train()

        #logger.info(f"Pretrain Start\t::\t{epoch+1} epoch")
        # loader는 epoch 시작할 때 마다 새로 만들어 줘야 그래프가 계속 생김
        loader = tqdm(train_generator, total = length)
        
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
            bbox_avg_loss += bbox_loss.mean()
            # Flow Data Train
            optimizer_flow.zero_grad()
            prediction = flow_model(flow_data)
            flow_loss = criterion(prediction, flow_data)
            flow_loss.mean().backward()
            optimizer_flow.step()
            flow_avg_loss += flow_loss.mean()
            # Ego Data Train
            optimizer_ego.zero_grad()
            prediction = ego_model(ego_data)
            ego_loss = criterion(prediction, ego_data)
            ego_loss.mean().backward()
            optimizer_ego.step()
            ego_avg_loss += ego_loss.mean()

            batch_number += bbox_data.shape[0]

        scheduler_bbox.step()
        scheduler_flow.step()
        scheduler_ego.step()

        epoch_time = time.time() - epoch_time
        logger.info(f'Pretrain AutoEncoder Training\t::\t{epoch+1}/{pretrain_epoch} :: Train Time :: {epoch_time:.3f}s '
                    f'BBox loss :: {bbox_avg_loss/batch_number:.6f} Flos loss :: {flow_avg_loss/batch_number:.6f} Ego Loss :: {ego_avg_loss/batch_number:.6f}')

        loader_val = tqdm(validation_generator, total = length_val)

        bbox_model.eval()
        flow_model.eval()
        ego_model.eval()
        criterion.eval()

        batch_number = 0
        epoch_time = time.time()
        
        bbox_avg_loss = 0
        flow_avg_loss = 0
        ego_avg_loss = 0
        with torch.no_grad():
            for data in loader_val:
                bbox_data, flow_data, ego_data = data
                prediction = bbox_model(bbox_data)
                bbox_loss = criterion(prediction, bbox_data)
                bbox_avg_loss += bbox_loss.mean()
                # Flow Data Train
                prediction = flow_model(flow_data)
                flow_loss = criterion(prediction, flow_data)
                flow_avg_loss += flow_loss.mean()
                # Ego Data Train
                prediction = ego_model(ego_data)
                ego_loss = criterion(prediction, ego_data)
                ego_avg_loss += ego_loss.mean()

                batch_number += bbox_data.shape[0]
        epoch_time = time.time() - epoch_time
        logger.info(f'Pretrain AutoEncoder Validation\t::\t{epoch+1}/{pretrain_epoch} :: Validation Time :: {epoch_time:.3f}s '
                    f'BBox loss :: {bbox_avg_loss/batch_number:.6f} Flos loss :: {flow_avg_loss/batch_number:.6f} Ego Loss :: {ego_avg_loss/batch_number:.6f}')
        
        ################################################
        # Callback 함수 들어갈 자기
        # 각 task별로 loss 계산한 다음에 validation loss가 이전보다 낮다면 해당 모델의 weight를 dictionary 타입으로 저장
        # 나중에 pretrained weight를 저장할 때 해당 dictioanry type weight을 load_state_dict를 통해서 업로드 한 다음에 모델을 저장
        ###############################################


    train_time = time.time() - train_time
    logger.info(f'Pretrain Finished\t::\t{train_time}')

    # 저장하기
    if not os.path.isdir(f'{result_path}pretrain/'):
        os.mkdir(f'{result_path}pretrain/')
    
    torch.save(bbox_model, f'{result_path}pretrain/bbox.pt')
    torch.save(flow_model, f'{result_path}pretrain/flow.pt')
    torch.save(ego_model, f'{result_path}pretrain/ego.pt')
    logger.info(f'Pretrain Weight Saved\t::\t{result_path}pretrain')

    bbox_model.to('cpu')
    flow_model.to('cpu')
    ego_model.to('cpu')
    return bbox_model, flow_model, ego_model, copy.deepcopy(ego_model)

def SADTrain(other_model, net_name):
    """
    1. 모델 gpu 이동
    2. 데이터 로더 불러오기 -> for_train 필요
    3. optimizer, scheduler 불러오기
    4. c 정보 모델에서 뽑기
    5. 자체 loss 수행
    """
    logger = logging.getLogger()
    logger.info(f'SAD Train Start ::\t{net_name}')
    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'
    other_model.to(device)

    optimizer_SAD = load_train_optimizer(other_model.parameters())
    schedular_SAD = load_train_multistep_lr(optimizer_SAD)
    
    start_time = time.time()
    

    train_generator, validation_generator = dataset_loader(net_name=net_name, state = 'SAD')

    center = other_model.c

    length = len(train_generator)
    length_val = len(validation_generator)
    logger.info(f'SAD Training Data Length\t::\t{length}')
    logger.info(f'SAD Validation Data Length\t::\t{length_val}')
    logger.info(f'SAD Training Epoch :: {train_epoch}')
    for epoch in range(train_epoch):
        other_model.train()

        loader = tqdm(train_generator, total = length)

        epoch_loss = 0.0
        n_batch = 0
        epoch_time = time.time()

        for data in loader:
            bbox, flow, ego, _, label, _ = data
            label = torch.unsqueeze(label, axis = -1)
            label = label.to(device)

            optimizer_SAD.zero_grad()
            result = other_model(bbox, flow, ego)
            distance = torch.sum((result - center) ** 2, dim = 1).squeeze()

            loss = torch.where(label == 1, distance, eta * ((distance + 1e-6) ** label.float()))
            loss = torch.mean(loss)
            loss.backward()
            optimizer_SAD.step()

            epoch_loss += loss.item()
            n_batch += bbox.shape[0]
        schedular_SAD.step()
        epoch_time = time.time() - epoch_time
        logger.info(f'Train SAD :: {epoch+1}/{train_epoch} :: Train Time :: {epoch_time:.3f}s '
                    f'loss :: {epoch_loss / n_batch:.6f}')
        
        loader_val = tqdm(validation_generator, total = length_val)

        other_model.eval()
        epoch_loss = 0.0
        n_batch = 0
        epoch_time = time.time()
        with torch.no_grad():
            for data in loader_val:
                bbox, flow, ego, _, label, _ = data
                label = torch.unsqueeze(label, axis = -1)
                label = label.to(device)
                result = other_model(bbox, flow, ego)
                distance = torch.sum((result - center) ** 2, dim = 1).squeeze()

                loss = torch.where(label == 1, distance, eta * ((distance + 1e-6) ** label.float()))
                loss = torch.mean(loss)

                epoch_loss += loss.item()
                n_batch += bbox.shape[0]
        epoch_time = time.time() - epoch_time
        logger.info(f'Validation SAD :: {epoch+1}/{train_epoch} :: Validation Time :: {epoch_time:.3f}s '
                    f'loss :: {epoch_loss / n_batch:.6f}')
        
        ######################################################################
        # 이하 동일


    start_time = time.time() - start_time
    logger.info(f'SAD Training Time :: {start_time:.3f}s')
    logger.info('SAD Training Finish')

    return other_model

            
def EGOTrain(ego_model, net_name):
    logger = logging.getLogger()
    logger.info(f'SAD_EGO Train Start ::\t{net_name}')
    device = 'cuda'
    if not torch.cuda.is_available():
        device = 'cpu'
    ego_model.to(device)

    optimizer_SAD = load_train_optimizer(ego_model.parameters())
    schedular_SAD = load_train_multistep_lr(optimizer_SAD)
    
    start_time = time.time()
    

    train_generator, validation_generator = dataset_loader(net_name=net_name, state = 'SAD_EGO')

    center = ego_model.c
    length = len(train_generator)
    length_val = len(validation_generator)
    logger.info(f'SAD_EGO Training Data Length :: {length}')
    logger.info(f'SAD_EGO Validation Data Length :: {length_val}')
    logger.info(f'SAD_EGO Training Epoch :: {train_epoch}')
    for epoch in range(train_epoch):
        ego_model.train()

        loader = tqdm(train_generator, total = length)
        
        epoch_loss = 0.0
        n_batch = 0
        epoch_time = time.time()

        for data in loader:
            ego, _, label = data
            label = torch.unsqueeze(label, axis = -1)
            label = label.to(device)

            optimizer_SAD.zero_grad()
            result = ego_model(ego)
            distance = torch.sum((result - center) ** 2, dim = 1).squeeze()
            loss = torch.where(label == 1, distance, eta * ((distance + 1e-6) ** label.float()))
            loss = torch.mean(loss)
            loss.backward()
            optimizer_SAD.step()

            epoch_loss += loss.item()
            n_batch += ego.shape[0]
        schedular_SAD.step()
        epoch_time = time.time() - epoch_time
        logger.info(f'Train SAD_EGO :: {epoch+1}/{train_epoch} :: Train Time :: {epoch_time:.3f}s '
                    f'loss :: {epoch_loss / n_batch:.6f}')
        
        loader_val = tqdm(validation_generator, total = length_val)

        ego_model.eval()

        epoch_loss = 0.0
        n_batch = 0
        epoch_time = time.time()

        with torch.no_grad():
            for data in loader_val:
                ego, _, label = data
                label = label.to(device)
                result = ego_model(ego)
                distance = torch.sum((result - center) ** 2, dim = 1).squeeze()
                loss = torch.where(label == 1, distance, eta * ((distance + 1e-6) ** label.float()))
                loss = torch.mean(loss)
                epoch_loss += loss.item()
                n_batch += ego.shape[0]
                
        epoch_time = time.time() - epoch_time
        logger.info(f'Validation SAD_EGO :: {epoch+1}/{train_epoch} :: Train Time :: {epoch_time:.3f}s '
                    f'loss :: {epoch_loss / n_batch:.6f}')



    start_time = time.time() - start_time
    logger.info(f'SAD_EGO Training Time :: {start_time:.3f}s')
    logger.info('SAD_EGO Training Finish')

    return ego_model