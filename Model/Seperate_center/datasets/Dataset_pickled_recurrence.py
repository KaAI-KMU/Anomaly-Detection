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
            self.all_inputs.append(os.path.join(self.data_root, name))
            #data = pkl.load(open(data_path, 'rb'))
            
            #input_bbox = data['bbox']
            #input_flow = data['flow']
            #input_ego = data['ego_motion']# [yaw, x, z]
            
            #self.all_inputs.append([input_bbox, input_flow, input_ego])
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        data = pkl.load(open(self.all_inputs[index], 'rb'))

        bbox = torch.FloatTensor(data['bbox']/255.0).to(device)
        flow = torch.FloatTensor(data['flow']/255.0).to(device)
        ego = torch.FloatTensor(data['ego_motion']/255.0).to(device)

        return bbox, flow, ego
    
class Recurrence_SAD_DATASET(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = f'{data_root}train/for_train/'
        self.data_name = os.listdir(self.data_root)

        self.all_inputs = []
        for name in self.data_name:
            # for each car in dataset, we split to several trainig samples
            
            #load data and label
            self.all_inputs.append(os.path.join(self.data_root, name)) # name = pickle file name -> all_inputs[index] = /workplace/Recurrence_data/train/for_train/{file_name}
            #data_path = os.path.join(self.data_root, name)
            #data = pkl.load(open(data_path, 'rb'))
            
            #input_bbox = data['bbox']
            #input_flow = data['flow']
            #input_ego = data['ego_motion']# [yaw, x, z]
            #video_name = '_'.join(name.split('_')[:2])
            #label = data['label']
            #frame_id = [data['frame_id']]
            
            #self.all_inputs.append([input_bbox, input_flow, input_ego, video_name, label, frame_id])
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        #input_bbox, input_flow, input_ego_motion, video_name, label, frame_id= self.all_inputs[index]
        #input_bbox = torch.FloatTensor(input_bbox/255).to(device)
        #input_flow = torch.FloatTensor(input_flow/255).to(device)
        #input_ego_motion = torch.FloatTensor(input_ego_motion/255).to(device)
        #frame_id = torch.IntTensor(frame_id)

        data = pkl.load(open(self.all_inputs[index], 'rb'))
        
        bbox = torch.FloatTensor(data['bbox']/255).to(device)
        flow = torch.FloatTensor(data['flow']/255).to(device)
        ego = torch.FloatTensor(data['ego_motion']/255).to(device)
        video_name = '_'.join(self.all_inputs[index].split('/')[-1].split('_')[:2])
        label = torch.IntTensor([data['label']]).to(device)
        frame_id = torch.IntTensor(data['frame_id']).to(device)

        return bbox, flow, ego, video_name, label, frame_id
    

class Recurrence_SAD_EGO_DATASET(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = f'{data_root}train/for_ego/'
        self.data_name = os.listdir(self.data_root)

        self.all_inputs = []
        for name in self.data_name:
            # for each car in dataset, we split to several trainig samples
            
            #load data and label
            
            self.all_inputs.append(os.path.join(self.data_root, name))
            
            #data_path = os.path.join(self.data_root, name)
            #data = pkl.load(open(data_path, 'rb'))#

            #input_ego = data['ego_motion']# [yaw, x, z]
            #video_name = '_'.join(name.split('_')[:2])
            #label = data['label']
            
            #self.all_inputs.append([input_ego, video_name, label])
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):

        data = pkl.load(open(self.all_inputs[index], 'rb'))

        ego = torch.FloatTensor(data['ego_motion']/255).to(device)
        video_name = '_'.join(self.all_inputs[index].split('/')[-1].split('_')[:2])
        label = torch.IntTensor([data['label']]).to(device)

        return ego, video_name, label
    