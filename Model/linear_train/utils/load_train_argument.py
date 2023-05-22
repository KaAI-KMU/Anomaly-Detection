from config.train_config import *
from torch import optim 
import torch.nn as nn

def load_pretrain_optimizer(parameters):
    if pretrain_optimzier == 'Adam':
        return optim.Adam(params=parameters, lr = pretrain_lr, weight_decay= pretrain_weight_decay)

def load_pretrain_multistep_lr(optimizer):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones = pretrain_milestone, gamma = pretrain_gamma)

def load_criterion():
    if pretrain_criterion == 'MSE':
        return nn.MSELoss(reduction='none')
    
def load_train_optimizer(parameters):
    if train_optimizer == 'Adam':
        return optim.Adam(params=parameters, lr = train_lr, weight_decay= train_weight_decay)
    
def load_train_multistep_lr(optimizer):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones = train_milestone, gamma = train_gamma)

