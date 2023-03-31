import torch
from torch.utils import data
import pickle as pkl
from config.data_config import *
from config.main_config import device
import os

if not torch.cuda.is_available():
    device = 'cpu'

class Recurrence_Pretrain_DATASET(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = f'{data_root}train/bdd/'
        self.data_name = os.listdir(self.data_root)

        self.all_inputs = []
        for name in self.data_name:
            # for each car in dataset, we split to several trainig samples
            
            #load data and label
            data_path = os.path.join(self.data_root, name)
            data = pkl.load(open(data_path, 'rb'))
            
            input_bbox = data['bbox']
            input_flow = data['flow']
            input_ego = data['ego_motion']# [yaw, x, z]
            
            self.all_inputs.append([input_bbox, input_flow, input_ego])
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_bbox, input_flow, input_ego_motion= self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox).to(device)
        input_flow = torch.FloatTensor(input_flow).to(device)
        input_ego_motion = torch.FloatTensor(input_ego_motion).to(device)

        return input_bbox, input_flow, input_ego_motion
    
class Recurrence_SAD_DATASET(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = f'{data_root}train/test/'
        self.data_name = os.listdir(self.data_root)

        self.all_inputs = []
        for name in self.data_name:
            # for each car in dataset, we split to several trainig samples
            
            #load data and label
            data_path = os.path.join(self.data_root, name)
            data = pkl.load(open(data_path, 'rb'))
            
            input_bbox = data['bbox']
            input_flow = data['flow']
            input_ego = data['ego_motion']# [yaw, x, z]
            video_name = '_'.join(name.split('_')[:2])
            
            self.all_inputs.append([input_bbox, input_flow, input_ego, video_name])
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_bbox, input_flow, input_ego_motion, video_name= self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox).to(device)
        input_flow = torch.FloatTensor(input_flow).to(device)
        input_ego_motion = torch.FloatTensor(input_ego_motion).to(device)

        return input_bbox, input_flow, input_ego_motion, video_name