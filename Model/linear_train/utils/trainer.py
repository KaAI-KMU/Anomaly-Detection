import torch
from model.network_builder import network_builder
import logging
from config.train_config import pretrain_optimzier, pretrain_lr, pretrain_weight_decay, pretrain_milestone, pretrain_gamma, pretrain_criterion, pretrain_epoch, train_optimizer, train_lr, train_weight_decay, train_milestone, train_gamma, train_epoch, eta
from config.main_config import device, network_name
from utils.load_train_argument import load_pretrain_optimizer, load_pretrain_multistep_lr, load_criterion, load_train_optimizer, load_train_multistep_lr
from utils.load_dataset import dataset_loader
from utils.callback_function import best_weight_callback
import time
from tqdm import tqdm
import copy
import os


def AETrain(net_name, result_path, feature):
    model = network_builder(net_name, feature, ego_only = False)
    logger = logging.getLogger()
    logger.info(f'Pretrain Start\t::\t{net_name}')

    optimizer = load_pretrain_optimizer(model.parameters())
    scheduler = load_pretrain_multistep_lr(optimizer)
    callback = best_weight_callback(name='model', center = False)

    criterion = load_criterion()

    model = model.to(device)

    criterion = criterion.to(device)

    train_time = time.time()
    train_generator, validation_generator = dataset_loader(net_name)
    length = len(train_generator)
    length_val = len(validation_generator)
    logger.info(f'Pretrain Training Data Length\t::\t{length}')
    logger.info(f'Pretrain Validation Data Length\t::\t{length_val}')
    logger.info(f'Pretrain Training Epoch\t::\t{pretrain_epoch}')
    for epoch in range(pretrain_epoch):
        model.train()
        criterion.train()

        #logger.info(f"Pretrain Start\t::\t{epoch+1} epoch")
        # loader는 epoch 시작할 때 마다 새로 만들어 줘야 그래프가 계속 생김
        loader = tqdm(train_generator, total = length)
        
        batch_number = 0
        epoch_time = time.time()
        
        avg_loss = 0       
        for data in loader:
            # Train
            optimizer.zero_grad()
            prediction = model(data)
            loss = criterion(prediction, data)
            loss.mean().backward()
            optimizer.step()
            avg_loss += loss.mean()

            batch_number += 1

        scheduler.step()

        epoch_time = time.time() - epoch_time
        logger.info(f'Pretrain AutoEncoder Training\t::\t{epoch+1}/{pretrain_epoch} :: Train Time :: {epoch_time:.3f}s '
                    f'loss :: {avg_loss/batch_number:.6f}')

        loader_val = tqdm(validation_generator, total = length_val)

        model.eval()
        criterion.eval()

        batch_number = 0
        epoch_time = time.time()
        
        avg_loss = 0
        with torch.no_grad():
            for data in loader_val:
                prediction = model(data)
                loss = criterion(prediction, data)
                avg_loss += loss.mean()
                
                batch_number += 1

        epoch_time = time.time() - epoch_time
        logger.info(f'Pretrain AutoEncoder Validation\t::\t{epoch+1}/{pretrain_epoch} :: Validation Time :: {epoch_time:.3f}s '
                    f'loss :: {avg_loss/batch_number:.6f}')
        
        ################################################
        # Callback 함수 들어갈 자기
        # 각 task별로 loss 계산한 다음에 validation loss가 이전보다 낮다면 해당 모델의 weight를 dictionary 타입으로 저장
        # 나중에 pretrained weight를 저장할 때 해당 dictioanry type weight을 load_state_dict를 통해서 업로드 한 다음에 모델을 저장
        ###############################################
        callback.add(model, avg_loss/batch_number)
        
    train_time = time.time() - train_time
    logger.info(f'Pretrain Finished\t::\t{train_time}')
    # Get Best model
    model = callback.get_best_model(model)
    # 저장하기
    if not os.path.isdir(f'{result_path}pretrain/'):
        os.mkdir(f'{result_path}pretrain/')
    
    torch.save(model, f'{result_path}pretrain/model.pt')
    logger.info(f'Pretrain Weight Saved\t::\t{result_path}pretrain')

    model.to('cpu')
    return model

def SADTrain(other_model, net_name, CALLBACK):
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
    if CALLBACK:
        callback_SAD = best_weight_callback(name = 'Other', center = True)
    
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

        for data, name, label, id in loader:

            #label = torch.unsqueeze(label, axis = -1)
            label = label.to(device)

            optimizer_SAD.zero_grad()
            result = other_model(data)
            distance = torch.sum((result.squeeze() - center) ** 2, dim = 1) # l1 loss

            loss = torch.zeros(data.shape[0]).to(device)
            for i in range(data.shape[0]):
                loss[i] = torch.where(label[i] == 0, distance[i], eta * ((distance[i] + 1e-6) ** (-1.0)))
            #loss = torch.where(label == 0, distance, eta * ((distance + 1e-6) ** (-1.0)))
            loss = torch.mean(loss)
            loss.backward()
            optimizer_SAD.step()

            epoch_loss += loss.item()
            n_batch += 1
        schedular_SAD.step()
        epoch_time = time.time() - epoch_time
        logger.info(f'Train SAD :: {epoch+1}/{train_epoch} :: Train Time :: {epoch_time:.3f}s '
                    f'loss :: {epoch_loss / n_batch:.6f}')
        
        loader_val = tqdm(validation_generator, total = length_val)

        tt, tf, ff, ft = 0,0,0,0

        other_model.eval()
        epoch_loss = 0.0
        n_batch = 0
        epoch_time = time.time()
        with torch.no_grad():
            for data, name, label, id in loader_val:
                #label = torch.unsqueeze(label, axis = -1)
                label = label.to(device)
                result = other_model(data)
                distance = torch.sum((result.squeeze() - center) ** 2, dim = 1).squeeze()

                for i in range(data.shape[0]):
                    if label[i] == 0 and distance[i] < 1:
                        tt += 1
                    elif label[i] == 0 and distance[i] >= 1:
                        tf += 1
                    elif label[i] == 1 and distance[i] >= 1:
                        ff += 1
                    elif label[i] == 1 and distance[i] < 1:
                        ft += 1

                loss = torch.zeros(data.shape[0]).to(device)
                for i in range(data.shape[0]):
                    loss[i] = torch.where(label[i] == 0, distance[i], eta * ((distance[i] + 1e-6) ** (-1.0)))
                #loss = torch.where(label == 0, distance, eta * ((distance + 1e-6) ** (-1.0)))
                loss = torch.mean(loss)
            
                epoch_loss += loss.item()
                n_batch += 1
        epoch_time = time.time() - epoch_time
        logger.info(f'Validation SAD :: {epoch+1}/{train_epoch} :: Validation Time :: {epoch_time:.3f}s '
                    f'loss :: {epoch_loss / n_batch:.6f} :: TT {tt}, TF {tf}, FF {ff}, FT {ft}')
        
        ######################################################################
        if CALLBACK:
            callback_SAD.add(other_model, epoch_loss / n_batch)


    start_time = time.time() - start_time
    logger.info(f'SAD Training Time :: {start_time:.3f}s')
    logger.info('SAD Training Finish')
    if CALLBACK:
        other_model = callback_SAD.get_best_model(other_model)

    return other_model