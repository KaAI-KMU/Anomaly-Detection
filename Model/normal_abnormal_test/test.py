import torch
from torch import nn
from utils.load_model import Load_model
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from config.data_config import *
from config.main_config import device
from config.test_config import *
import os

class Recurrence_SAD_DATASET(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = f'{data_root}val/for_train/'
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
            
    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_bbox, input_flow, input_ego_motion, video_name, label, frame_id= self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox/255).to(device)
        input_flow = torch.FloatTensor(input_flow/255).to(device)
        input_ego_motion = torch.FloatTensor(input_ego_motion/255).to(device)
        frame_id = torch.IntTensor(frame_id)

        return input_bbox, input_flow, input_ego_motion, video_name, label, frame_id
    

class Recurrence_SAD_EGO_DATASET(data.Dataset):
    def __init__(self):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.data_root = f'{data_root}val/for_ego/'
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
    
other, ego = Load_model(weight_path)

other = other.to('cuda')
ego_model = ego.to('cuda')

other_dataset = data.DataLoader(Recurrence_SAD_DATASET(), batch_size=data_batch, shuffle = True, num_workers=0)

normal = []
abnormal = []

center = other.c

with torch.no_grad():
    for data in other_dataset:
        bbox, flow, ego, _, label, _ = data
        result = other(bbox, flow, ego)
        distance = torch.sum((result.squeeze() - center) ** 2, dim = 1).squeeze()

        for index in range(bbox.shape[0]):
            if label[index] == 0:
                normal.append(distance[index].item())
            else:
                abnormal.append(distance[index].item())

normal = np.array(normal)
abnormal = np.array(abnormal)

cross = np.mean(normal)

tf = 0
ft = 0

tt = 0
ff = 0

t_count = 0
f_count = 0

for i in normal:
    if i > cross:
        tf += 1
    else:
        tt += 1
    t_count += 1

for i in abnormal:
    if i < cross:
        ft += 1
    else:
        ff += 1
    f_count += 1

print(tt)
print(tf)
print(ff)
print(ft)

print(tt/t_count)
print(tf/t_count)
print(ff/f_count)
print(ft/f_count)

print((tt+ff)/(t_count + f_count))

train_dataset = data.DataLoader(Recurrence_SAD_EGO_DATASET(), batch_size=32, shuffle=True, num_workers=0)

normal = []
abnormal = []

center = ego_model.c

with torch.no_grad():
    for data in train_dataset:
        ego, _, label = data
        result = ego_model(ego)
        distance = torch.sum((result.squeeze() - center) ** 2, dim = 1).squeeze()

        for index in range(bbox.shape[0]):
            if label[index] == 0:
                normal.append(distance[index].item())
            else:
                abnormal.append(distance[index].item())


normal = np.array(normal)
abnormal = np.array(abnormal)

cross = np.mean(normal)

tf = 0
ft = 0

tt = 0
ff = 0

t_count = 0
f_count = 0

for i in normal:
    if i > cross:
        tf += 1
    else:
        tt += 1
    t_count += 1

for i in abnormal:
    if i < cross:
        ft += 1
    else:
        ff += 1
    f_count += 1

print(tt)
print(tf)
print(ff)
print(ft)

print(tt/t_count)
print(tf/t_count)
print(ff/f_count)
print(ft/f_count)

print((tt+ff)/(t_count + f_count))