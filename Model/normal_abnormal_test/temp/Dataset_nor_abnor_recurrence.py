import torch
from torch.utils import data
import pickle as pkl
from config.data_config import *
from config.main_config import device
import os


class Recurrence_SAD_DATASET_NORMAL(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = f'/workspace/SEPERATED_RECURRENCE_DATA/train/for_train_nor'
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
            label = data['label']
            frame_id = [data['frame_id']]
            
            self.all_inputs.append([input_bbox, input_flow, input_ego, video_name, label, frame_id])
            if len(self.all_inputs) == 100:
                break
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_bbox, input_flow, input_ego_motion, video_name, label, frame_id= self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox/255).to(device)
        input_flow = torch.FloatTensor(input_flow/255).to(device)
        input_ego_motion = torch.FloatTensor(input_ego_motion/255).to(device)
        frame_id = torch.IntTensor(frame_id).to(device)

        return input_bbox, input_flow, input_ego_motion, video_name, label, frame_id
    

class Recurrence_SAD_DATASET_ABNORMAL(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = f'/workspace/SEPERATED_RECURRENCE_DATA/train/for_train_abnor'
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
            label = data['label']
            frame_id = [data['frame_id']]
            
            self.all_inputs.append([input_bbox, input_flow, input_ego, video_name, label, frame_id])
            if len(self.all_inputs) == 100:
                break
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_bbox, input_flow, input_ego_motion, video_name, label, frame_id= self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox/255).to(device)
        input_flow = torch.FloatTensor(input_flow/255).to(device)
        input_ego_motion = torch.FloatTensor(input_ego_motion/255).to(device)
        frame_id = torch.IntTensor(frame_id).to(device)

        return input_bbox, input_flow, input_ego_motion, video_name, label, frame_id
    




class Recurrence_SAD_EGO_DATASET_NORMAL(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = '/workspace/SEPERATED_RECURRENCE_DATA/train/for_ego_nor'
        self.data_name = os.listdir(self.data_root)

        self.all_inputs = []
        for name in self.data_name:
            # for each car in dataset, we split to several trainig samples
            
            #load data and label
            data_path = os.path.join(self.data_root, name)
            data = pkl.load(open(data_path, 'rb'))

            input_ego = data['ego_motion']# [yaw, x, z]
            video_name = '_'.join(name.split('_')[:2])
            label = data['label']
            
            self.all_inputs.append([input_ego, video_name, label])
            if len(self.all_inputs) == 100:
                break
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_ego_motion, video_name, label= self.all_inputs[index]
        input_ego_motion = torch.FloatTensor(input_ego_motion/255).to(device)

        return input_ego_motion, video_name, label
    

class Recurrence_SAD_EGO_DATASET_ABNORMAL(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = '/workspace/SEPERATED_RECURRENCE_DATA/train/for_ego_abnor'
        self.data_name = os.listdir(self.data_root)

        self.all_inputs = []
        for name in self.data_name:
            # for each car in dataset, we split to several trainig samples
            
            #load data and label
            data_path = os.path.join(self.data_root, name)
            data = pkl.load(open(data_path, 'rb'))

            input_ego = data['ego_motion']# [yaw, x, z]
            video_name = '_'.join(name.split('_')[:2])
            label = data['label']
            
            self.all_inputs.append([input_ego, video_name, label])
            if len(self.all_inputs) == 100:
                break
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_ego_motion, video_name, label= self.all_inputs[index]
        input_ego_motion = torch.FloatTensor(input_ego_motion/255).to(device)

        return input_ego_motion, video_name, label