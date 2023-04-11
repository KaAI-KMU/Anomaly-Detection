import os
from recurrence import rec_plot
import pickle as pkl


data_length = 16
stride = 1

flow_eps = [0.01, 0.009]
bbox_eps = [0.0003, 0.0005, 0.0005, 0.0006]
ego_eps = [0.01, 0.01, 0.01]

BASE_PATH = 'D:/DoTA/DATA/'
SAVE_PATH = 'C:/MT/KaAI/GitHub/Model/New_train/RECURRENCE_DATA/'
state = 'train'


base_path = f'{BASE_PATH}{state}/'
save_path = f'{SAVE_PATH}{state}/'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

folder_names = os.listdir(base_path)

for folder_name in folder_names:
    specific_path = f'{base_path}{folder_name}/'
    specific_save_path = f'{save_path}{folder_name}/'
    if not os.path.isdir(specific_save_path): # bdd
        os.mkdir(specific_save_path)
    pickle_names = os.listdir(specific_path)
    for pickle_name in pickle_names:
        LABEL = False
        FLOW = False
        FRAME = False
        pickle_load_path = f'{specific_path}{pickle_name}'
        data = pkl.load(open(pickle_load_path, 'rb'))

        ego_motion = data['ego_motion']
        if 'label' in data.keys():
            label = data['label']
            LABEL = True
        if 'bbox' in data.keys():
            bbox = data['bbox']
            flow = data['flow']
            FLOW = True
        if 'frame_id' in data.keys():
            frame = data['frame_id']
            FRAME = True

        for start in range(0, len(ego_motion) - data_length, stride):
            if FLOW:
                input_bbox = rec_plot(bbox[start:start+data_length], bbox_eps)
                input_flow = rec_plot(flow[start:start+data_length], flow_eps, flow = True)
            input_ego = rec_plot(ego_motion[start:start+data_length], ego_eps)
            if LABEL:
                if sum(label[start:start+data_length]) != 0:
                    temp_label = 1
                else:
                    temp_label = 0
            temp = {'ego_motion' : input_ego}
            if FLOW:
                temp['bbox'] = input_bbox
                temp['flow'] = input_flow
            if LABEL:
                temp['label'] = temp_label
            if FRAME:
                temp['frame_id'] = frame[start:start+data_length]
            
            with open(f'{specific_save_path}{pickle_name.split(".")[0]}_{str(start)}.pkl', 'wb') as f:
                pkl.dump(temp, f)

data_length = 16
stride = 1

flow_eps = [0.01, 0.009]
bbox_eps = [0.0003, 0.0005, 0.0005, 0.0006]
ego_eps = [0.01, 0.01, 0.01]

BASE_PATH = 'D:/DoTA/DATA/'
SAVE_PATH = 'C:/MT/KaAI/GitHub/Model/New_train/RECURRENCE_DATA/'
state = 'val'

base_path = f'{BASE_PATH}{state}/'
save_path = f'{SAVE_PATH}{state}/'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

folder_names = os.listdir(base_path)

for folder_name in folder_names:
    specific_path = f'{base_path}{folder_name}/'
    specific_save_path = f'{save_path}{folder_name}/'
    if not os.path.isdir(specific_save_path): # bdd
        os.mkdir(specific_save_path)
    pickle_names = os.listdir(specific_path)
    for pickle_name in pickle_names:
        LABEL = False
        FLOW = False
        FRAME = False
        pickle_load_path = f'{specific_path}{pickle_name}'
        data = pkl.load(open(pickle_load_path, 'rb'))

        ego_motion = data['ego_motion']
        if 'label' in data.keys():
            label = data['label']
            LABEL = True
        if 'bbox' in data.keys():
            bbox = data['bbox']
            flow = data['flow']
            FLOW = True
        if 'frame_id' in data.keys():
            frame = data['frame_id']
            FRAME = True

        for start in range(0, len(ego_motion) - data_length, stride):
            if FLOW:
                input_bbox = rec_plot(bbox[start:start+data_length], bbox_eps)
                input_flow = rec_plot(flow[start:start+data_length], flow_eps, flow = True)
            input_ego = rec_plot(ego_motion[start:start+data_length], ego_eps)
            if LABEL:
                if sum(label[start:start+data_length]) != 0:
                    temp_label = 1
                else:
                    temp_label = 0
            temp = {'ego_motion' : input_ego}
            if FLOW:
                temp['bbox'] = input_bbox
                temp['flow'] = input_flow
            if LABEL:
                temp['label'] = temp_label
            if FRAME:
                temp['frame_id'] = frame[start:start+data_length]
            
            with open(f'{specific_save_path}{pickle_name.split(".")[0]}_{str(start)}.pkl', 'wb') as f:
                pkl.dump(temp, f)