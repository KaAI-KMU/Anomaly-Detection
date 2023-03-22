import torch
from torch.utils import data
import pickle as pkl
from utils.recurrence import rec_plot
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
            
            bbox = data['bbox']
            flow = data['flow']
            ego_motion = data['ego_motion']# [yaw, x, z]
            
            # frame_id = data['frame_id']

            # go farwad along the session to get data samples
            for start in range(0, len(flow) - data_length, stride):
                
                input_bbox = rec_plot(bbox[start:start+data_length])
                input_flow = rec_plot(flow[start:start+data_length], flow = True)
                input_ego = rec_plot(ego_motion[start:start+data_length])
                self.all_inputs.append([input_bbox, input_flow, input_ego])
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_bbox, input_flow, input_ego_motion= self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox).to(device)
        input_flow = torch.FloatTensor(input_flow).to(device)
        input_ego_motion = torch.FloatTensor(input_ego_motion).to(device)

        return input_bbox, input_flow, input_ego_motion