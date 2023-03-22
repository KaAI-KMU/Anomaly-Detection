from config.main_config import *
from torch import optim 
import torch.nn as nn

def load_optimizer(parameters):
    if pretrain_optimzier == 'Adam':
        return optim.Adam(params=parameters, lr = pretrain_lr, weight_decay= pretrain_weight_decay)

def laod_multistep_lr(optimizer):
    return optim.lr_scheduler.MultiplicativeLR(optimizer, milestones = pretrain_milestone, gamma = pretrain_gamma)

def load_criterion():
    if pretrain_criterion == 'MSE':
        return nn.MSELoss(reduction='none')