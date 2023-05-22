import torch
from torch.utils import data
import pickle as pkl
from config.data_config import *
from config.main_config import device
import os
import numpy as np

if not torch.cuda.is_available():
    device = 'cpu'

THRESHOLD = 10

class Linear_Pretrain_DATASET(data.Dataset):
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

            self.all_inputs.append(os.path.join(self.data_root, name))

            #if len(self.all_inputs) == 1000:
            #    break

            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        data = pkl.load(open(self.all_inputs[index], 'rb'))

        input_bbox = data['bbox']

        # Flow :: Horizontal Flow, Vertical Flow -> 범위가 -400 ~ 400 이다
        input_flow = data['flow']
        # -400 ~ 400 사이의 범위로 변환
        input_flow[:,0] = np.where(input_flow[:,0] < THRESHOLD, input_flow[:,0], THRESHOLD)
        input_flow[:,0] = np.where(input_flow[:,0] > -THRESHOLD, input_flow[:,0], -THRESHOLD)
        # 0 ~ 800 사이의 범위로 변환한 다음 800으로 나누어서 0 ~ 1 사이의 범위로 변환
        input_flow [:,0] = (input_flow[:,0] + THRESHOLD) / (THRESHOLD*2)
        # -400 ~ 400 사이의 범위로 변환
        input_flow[:,1] = np.where(input_flow[:,1] < THRESHOLD, input_flow[:,1], THRESHOLD)
        input_flow[:,1] = np.where(input_flow[:,1] > -THRESHOLD, input_flow[:,1], -THRESHOLD)
        # 0 ~ 800 사이의 범위로 변환한 다음 800으로 나누어서 0 ~ 1 사이의 범위로 변환
        input_flow [:,1] = (input_flow[:,1] + THRESHOLD) / (THRESHOLD*2)

        input_ego = data['ego_motion']
        input_ego[:,0] = (input_ego[:,0] + np.math.pi) / (2*np.math.pi)
        input_ego[:,1:] = np.abs(input_ego[:,1:] - np.vstack((np.array([input_ego[0,1:]]), input_ego[:-1,1:])))

        input_bbox = (torch.FloatTensor(input_bbox))
        input_flow = (torch.FloatTensor(input_flow))
        input_ego = (torch.FloatTensor(input_ego))

        return torch.cat((input_bbox, input_flow, input_ego), axis = -1).to(device)
    
class Linear_SAD_DATASET(data.Dataset):
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

            self.all_inputs.append(os.path.join(self.data_root, name))
            #data_path = os.path.join(self.data_root, name)
            #data = pkl.load(open(data_path, 'rb'))
            
            #input_bbox = data['bbox']
            #input_flow = data['flow']
            #input_ego = data['ego_motion']# [yaw, x, z]
            #video_name = '_'.join(name.split('_')[:2])
            #label = data['label']
            #frame_id = [data['frame_id']]
            
            #self.all_inputs.append([input_bbox, input_flow, input_ego, video_name, label, frame_id])
            #if len(self.all_inputs) == 1000:
            #    break
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        data = pkl.load(open(self.all_inputs[index], 'rb'))

        input_bbox = data['bbox']

        # Flow :: Horizontal Flow, Vertical Flow -> 범위가 -400 ~ 400 이다
        input_flow = data['flow']
        # -400 ~ 400 사이의 범위로 변환
        input_flow[:,0] = np.where(input_flow[:,0] < THRESHOLD, input_flow[:,0], THRESHOLD)
        input_flow[:,0] = np.where(input_flow[:,0] > -THRESHOLD, input_flow[:,0], -THRESHOLD)
        # 0 ~ 800 사이의 범위로 변환한 다음 800으로 나누어서 0 ~ 1 사이의 범위로 변환
        input_flow [:,0] = (input_flow[:,0] + THRESHOLD) / (THRESHOLD*2)
        # -400 ~ 400 사이의 범위로 변환
        input_flow[:,1] = np.where(input_flow[:,1] < THRESHOLD, input_flow[:,1], THRESHOLD)
        input_flow[:,1] = np.where(input_flow[:,1] > -THRESHOLD, input_flow[:,1], -THRESHOLD)
        # 0 ~ 800 사이의 범위로 변환한 다음 800으로 나누어서 0 ~ 1 사이의 범위로 변환
        input_flow [:,1] = (input_flow[:,1] + THRESHOLD) / (THRESHOLD*2)

        input_ego = data['ego_motion']
        input_ego[:,0] = (input_ego[:,0] + np.math.pi) / (2*np.math.pi)
        input_ego[:,1:] = np.abs(input_ego[:,1:] - np.vstack((np.array([input_ego[0,1:]]), input_ego[:-1,1:])))
        
        video_name = '_'.join(self.all_inputs[index].split('/')[-1].split('_')[:2])
        label = torch.IntTensor([data['label']])
        frame_id = torch.IntTensor([data['frame_id']])

        input_bbox = (torch.FloatTensor(input_bbox))
        input_flow = (torch.FloatTensor(input_flow))
        input_ego = (torch.FloatTensor(input_ego))

        return torch.cat((input_bbox, input_flow, input_ego),axis = -1).to(device), video_name, label, frame_id


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
            data_path = os.path.join(self.data_root, name)
            data = pkl.load(open(data_path, 'rb'))

            input_ego = data['ego_motion']# [yaw, x, z]
            video_name = '_'.join(name.split('_')[:2])
            label = data['label']
            
            self.all_inputs.append([input_ego, video_name, label])
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_ego_motion, video_name, label= self.all_inputs[index]
        input_ego_motion = torch.FloatTensor(input_ego_motion/255).to(device)

        return input_ego_motion, video_name, label
    