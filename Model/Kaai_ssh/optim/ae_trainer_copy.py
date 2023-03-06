import torch
from torch.nn import functional as F
from tqdm import tqdm
import logging
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.args import parse_args
from datasets.Dataset import DATASET, PRETRAIN_DATASET
from datasets.Dataset_recurrence import Recurrence_DATASET
from config.main_config import ae_lr_milestone
from torch.utils.data import DataLoader 


class AETrainer(nn.Module):
    
    def __init__(self, net_name) -> None:
        super().__init__()

        # Results
        self.train_time = None
        self.validartion_loss = {'BBox' : None, 'Flow' : None, 'Ego' : None}
        self.validation_time = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = parse_args()
        self.dataloader_params ={
            "batch_size": self.args['batch_size'],
            "shuffle": self.args['shuffle'],
            "num_workers": self.args['num_workers']
            }
        self.net_name = net_name
        
    def train(self, ae_flow, ae_bbox, ae_ego):

        args = parse_args()
        dataloader_params ={
        "batch_size": args['batch_size'],
        "shuffle": args['shuffle'],
        "num_workers": args['num_workers']
    }
        logger = logging.getLogger()
        
        optimizer_bbox = optim.Adam(ae_bbox.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler_bbox = optim.lr_scheduler.MultiStepLR(optimizer_bbox, milestones=ae_lr_milestone , gamma=0.1)

        optimizer_flow = optim.Adam(ae_flow.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler_flow = optim.lr_scheduler.MultiStepLR(optimizer_flow, milestones=ae_lr_milestone ,gamma=0.1)

        optimizer_ego = optim.Adam(ae_ego.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler_ego = optim.lr_scheduler.MultiStepLR(optimizer_ego,milestones=ae_lr_milestone , gamma=0.1)
        
        ae_bbox = ae_bbox.to(self.device)
        ae_flow = ae_flow.to(self.device)
        ae_ego = ae_ego.to(self.device)

        criterion = nn.MSELoss(reduction='none')
        criterion = criterion.to(self.device)
        
        start_time = time.time()
        epochs = self.args['nb_fol_epoch']

        for epoch in range(epochs):
            ae_flow.train()
            ae_bbox.train()
            ae_ego.train()
            
            if self.net_name == 'GRU_TAD':
                train_dataset = PRETRAIN_DATASET(args, 'train')
            elif self.net_name == 'Recurrence':
                train_dataset = Recurrence_DATASET(args, 'train')

            train_gen = DataLoader(train_dataset, **dataloader_params)     
            loader = tqdm(train_gen, total = len(train_gen))
            logger.info(f'Starting pretraining :: Train Epoch {epoch}...')
            
            ae_bbox.train()
            ae_flow.train()
            ae_ego.train()

            n_batches = 0
            epoch_start_time = time.time()
            for data in loader:
                bbox_in, flow_in, ego_in, bbox_out, flow_out, ego_out = data
                # 입력 3개, 출력3개, 라벨
                
                optimizer_bbox.zero_grad() 
                prediction = ae_bbox(bbox_in)
                bbox_loss = criterion(prediction, bbox_out)
                bbox_loss = torch.mean(bbox_loss)
                bbox_loss.backward()

                optimizer_flow.zero_grad()
                prediction = ae_flow(flow_in)
                if self.net_name == 'GRU_TAD':
                    flow_out = flow_out[:,flow_out.shape[1]//2, flow_out.shape[2]//2, :]
                flow_loss = criterion(prediction, flow_out)
                flow_loss = torch.mean(flow_loss)
                flow_loss.backward()

                optimizer_ego.zero_grad() 
                prediction = ae_ego(ego_in)
                ego_loss = criterion(prediction, ego_out)
                ego_loss = torch.mean(ego_loss)
                ego_loss.backward()

                optimizer_bbox.step()
                optimizer_flow.step()
                optimizer_ego.step()
                
                n_batches += 1

            scheduler_bbox.step()
            scheduler_flow.step()
            scheduler_ego.step()
                
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'Train | Epoch: {epoch + 1:03}/{epochs:03} | Train Time: {epoch_train_time:.3f}s '
                            f'| BBox Loss: {bbox_loss / n_batches:.6f} | Flow Loss: {flow_loss / n_batches:.6f} | Ego Loss: {ego_loss / n_batches:.6f}')


        self.train_time = time.time() - start_time
        logger.info('Pretraining Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished pretraining.')

        return ae_bbox, ae_flow, ae_ego
    
    def validation(self, ae_bbox, ae_flow, ae_ego):

        criterion = nn.MSELoss(reduction='none')
        criterion = criterion.to(self.device)
        
        args = parse_args()
        dataloader_params ={
        "batch_size": args['batch_size'],
        "shuffle": args['shuffle'],
        "num_workers": args['num_workers']
    }
        logger = logging.getLogger()
        
        if self.net_name == 'GRU_TAD':
            val_dataset = PRETRAIN_DATASET(args, 'val')
        elif self.net_name == 'Recurrence':
            val_dataset = Recurrence_DATASET(args, 'val')

        val_gen = DataLoader(val_dataset, **dataloader_params)  
        loader = tqdm(val_gen, total = len(val_gen))

        ae_flow.eval()
        ae_bbox.eval()
        ae_ego.eval()

        logger.info(f'Starting pretraining :: Evaluation')

        eval_batch = 0

        start = time.time()
        for data in loader:
            bbox_in, flow_in, ego_in, bbox_out, flow_out, ego_out = data
            # 입력 3개, 출력3개, 라벨
            
            prediction = ae_bbox(bbox_in)
            bbox_loss_eval = criterion(prediction, bbox_out)
            bbox_loss_eval = torch.mean(bbox_loss_eval)

            prediction = ae_flow(flow_in)
            if self.net_name == 'GRU_TAD':
                flow_out = flow_out[:,flow_out.shape[1]//2, flow_out.shape[2]//2, :]
            flow_loss_eval = criterion(prediction, flow_out)
            flow_loss_eval = torch.mean(flow_loss_eval)

            prediction = ae_ego(ego_in)
            ego_loss_eval = criterion(prediction, ego_out)
            ego_loss_eval = torch.mean(ego_loss_eval)

            eval_batch += 1

        self.validation_time = start - time.time()

        logger.info(f'Validation | Validation Time :: {self.validation_time:.3f}s'
                        f'| BBox Loss: {bbox_loss_eval / eval_batch:.6f} | Flow Loss: {flow_loss_eval / eval_batch:.6f} | Ego Loss: {ego_loss_eval / eval_batch:.6f}')
        
        self.validartion_loss['BBox'] = (bbox_loss_eval / eval_batch).item()
        self.validartion_loss['Flow'] =(flow_loss_eval / eval_batch).item()
        self.validartion_loss['Ego'] = (ego_loss_eval / eval_batch).item()


    def test(self, model):

        logger = logging.getLogger()
        
        val = DATASET(self.args, 'val')
        val_gen = DataLoader(val, **self.dataloader_params)
        
        # Get test data loader
        loader = tqdm(val_gen, total = len(val_gen))

        #Set loss
        criterion = nn.MSELoss(reduction='none')

        # Set device for network
        ae_net = model.to(self.device)
        criterion = criterion.to(self.device)

        # Testing
        logger.info('Testing autoencoder...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        ae_net.eval()

        with torch.no_grad():
            for data in loader:
                bbox_in, flow_in, _, bbox_out, _ = data
                
                prediction = ae_net(flow_in, bbox_in)
                loss = criterion(prediction, bbox_out)
                loss = torch.mean(loss)
                epoch_loss += loss.item()
                n_batches += 1
                
        test_time = time.time() - start_time

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * test_auc))
        logger.info('Test Time: {:.3f}s'.format(test_time))
        logger.info('Finished testing autoencoder.')