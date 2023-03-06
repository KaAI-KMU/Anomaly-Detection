import numpy as np
import torch
from config.model_config import * 
import os
import pickle as pkl
import glob
from torch.utils import data
import json


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rec_plot_module(s, eps=0.01, steps = 255):
    N = s.size
    S = np.repeat(s[None,:], N, axis = 0)
    Z = np.floor(np.abs(S-S.T)/eps)
    Z[Z>steps] = steps
    
    return Z

def rec_plot(s, flow = False):
    if flow:
        storage = np.zeros((s.shape[0], 2))
        for index in range(storage.shape[0]):
            storage[index,:] = s[index, s.shape[1]//2, s.shape[2]//2, :]
        s = storage
    storage = np.zeros((s.shape[1], s.shape[0], s.shape[0]))
    for index in range(s.shape[1]):
        storage[index, :, :] = rec_plot_module(s[:, index])
    return storage

class Recurrence_DATASET(data.Dataset):
    def __init__(self, args, phase):
        '''
        HEV-I dataset object. Contains bbox, flow and ego motion.
        
        Params:
            args: arguments passed from main file
            phase: 'train' or 'val'
        '''
        self.args = args
        self.data_root = os.path.join(self.args['data_root'], phase)
        self.video_name = os.listdir(self.data_root)
        
        self.sessions = glob.glob(os.path.join(self.data_root,'*'))

        self.all_inputs = []
        for name in self.video_name:
            # for each car in dataset, we split to several trainig samples
            
            #load data and label
            data_path = os.path.join(self.data_root, name)
            try:
                label_path = os.path.join(self.label_root, name + '.json')
                label = json.load(open(label_path, 'rb'))
                labels = label["labels"] #[frame]['accident_id']
            except:
                labels = None
            data = pkl.load(open(data_path, 'rb'))
            
            bbox = data['bbox']
            flow = data['flow']
            ego_motion = data['ego_motion']# [yaw, x, z]
            
            # frame_id = data['frame_id']

            # go farwad along the session to get data samples
            seed = np.random.randint(self.args['seed_max'])
            for start in range(seed, len(flow), int(self.args['segment_len']/2)):
                end = start + self.args['segment_len']
                if end + self.args['pred_timesteps'] <= len(bbox) and end <= len(flow):
                    input_bbox = rec_plot(bbox[start:end,:])
                    input_flow = rec_plot(flow[start:end,:,:,:], flow = True)
                    input_ego_motion = rec_plot(ego_motion[start:end,:])
                    # target_ego_motion = self.get_target(ego_motion_session, ego_start, ego_end)
                    # if input_flow.shape[0] != 16:
                    #     print(flow.shape)
                    #     print(bbox.shape)
                    #     print(input_flow.shape)
                    #     print("start: {} end:{} length:{}".format(start, end, self.args.segment_len))
                    
                    self.all_inputs.append([input_bbox, input_flow, input_ego_motion])
            
            # go backward along the session to get data samples again
            seed = np.random.randint(self.args['seed_max'])
            for end in range(min([len(bbox)-self.args['pred_timesteps'], len(flow)]), 
                             seed, 
                             -self.args['segment_len']):

                start = end - self.args['segment_len']
                if start >= 0:
                    input_bbox = rec_plot(bbox[start:end,:])
                    input_flow = rec_plot(flow[start:end,:,:,:], flow = True)
                    input_ego_motion = rec_plot(ego_motion[start:end, :])
                    
                    # if input_flow.shape[0] != 16:
                    #     print(flow.shape)
                    #     print(bbox.shape)
                    #     print(input_flow.shape)
                    #     print("start: {} end:{} length:{}".format(start, end, self.args.segment_len))
                    self.all_inputs.append([input_bbox, input_flow, input_ego_motion])

    def __len__(self):
        return len(self.all_inputs)
    
    def __getitem__(self, index):
        input_bbox, input_flow, input_ego_motion= self.all_inputs[index]
        input_bbox = torch.FloatTensor(input_bbox).to(device)
        input_flow = torch.FloatTensor(input_flow).to(device)
        input_ego_motion = torch.FloatTensor(input_ego_motion).to(device)

        return input_bbox, input_flow, input_ego_motion, input_bbox, input_flow, input_ego_motion