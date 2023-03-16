from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from datasets.Dataset import DATASET

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import tqdm
from utils.args import parse_args

from config.main_config import *


class DeepSADTrainer():

    def __init__(self, c):
        super().__init__()

        # Deep SAD parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.c = torch.tensor(c, device=self.device) if c is not None else None

        self.eta = eta
        self.lr = lr
        self.epochs = n_epochs
        self.lr_milestones = lr_milestone
        self.batch_size = batch_size
        self.n_jobs_dataloader = n_jobs_dataloader
        
        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None
        
        self.args = parse_args()

    def train(self, model):
        logger = logging.getLogger()

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestone, gamma=0.1)
        
        # Set device for network
        net = model.to(self.device)

        epochs = self.args['nb_fol_epoch']
        
        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            train = DATASET(self.args, 'train')
            train_gen = data.DataLoader(train, **self.dataloader_params)
            train_loader = tqdm(train_gen, total = len(train_gen))
            self.c = self.init_center_c(train_loader, net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(epochs):
            
            scheduler.step()
            
            epoch_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                bbox_in, flow_in, _, bbox_out, _, cls = data #ego_in, ego_out
                
                # abnormal with ego 일 경우 abnormal로 취급
                if cls == -2.0:
                    cls = -1.0
                
                bbox_in, flow_in, cls = bbox_in.to(self.device), flow_in.to(self.device), cls.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                outputs = net(bbox_in, flow_in)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(cls == 0, dist, self.eta * ((dist + self.eps) ** cls.float()))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return net

    def test(self, net):
        logger = logging.getLogger()
        test = DATASET(self.args, 'test')
        val_gen = data.DataLoader(test, **self.dataloader_params)
        # Get test data loader
        loader = tqdm(val_gen, total = len(val_gen))

        # Set device for network
        net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in loader:
                bbox_in, flow_in, _, bbox_out, _, cls = data
                
                # abnormal with ego 일 경우 abnormal로 취급
                if cls == -2.0:
                    cls = -1.0

                bbox_in = bbox_in.to(self.device)
                flow_in = flow_in.to(self.device)
                bbox_out = bbox_out.to(self.device)
                cls = cls.to(self.device)
                
                outputs = net(bbox_in, flow_in)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)
                losses = torch.where(cls == 0, dist, self.eta * ((dist + self.eps) ** cls.float()))
                loss = torch.mean(losses)
                scores = dist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(cls.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(labels, scores)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                bbox_in, flow_in, ego, _, _, _ = data
                outputs = net(bbox_in, flow_in, ego)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
