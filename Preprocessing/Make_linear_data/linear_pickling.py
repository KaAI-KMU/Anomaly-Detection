import os
import pickle as pkl
import numpy as np

data_length = 16
stride = 1

BASE_PATH = '/media/kaai/MT/DoTA/DATA/'
SAVE_PATH = '/home/kaai/MT/LINEAR_DATA/'
state = 'train'

count = 0

base_path = f'{BASE_PATH}{state}/'
save_path = f'{SAVE_PATH}{state}/'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

folder_names = os.listdir(base_path)
print('Train Session Start')
for folder_name in folder_names:
    print(f'{folder_name} Data Start')
    specific_path = f'{base_path}{folder_name}/'
    specific_save_path = f'{save_path}{folder_name}/'
    if not os.path.isdir(specific_save_path): # bdd
        os.mkdir(specific_save_path)
    pickle_names = os.listdir(specific_path)
    for index, pickle_name in enumerate(pickle_names):

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
        if index % 100 == 0 and (len(ego_motion) - data_length)//stride > 0:
            print(f'{pickle_name} data start')
            print(f'Possible amount of data :: {(len(ego_motion) - data_length)//stride}')
            print('')
        for start in range(0, len(ego_motion) - data_length, stride):
            if FLOW:
                input_bbox = np.array(bbox[start:start+data_length])
                input_flow = np.array(flow[start:start+data_length])
                input_flow = input_flow[:, input_flow.shape[1]//2, input_flow.shape[2]//2]
            input_ego = np.array(ego_motion[start:start+data_length])
            
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
                count += 1
        

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
                input_bbox = np.array(bbox[start:start+data_length])
                input_flow = np.array(flow[start:start+data_length])
                input_flow = input_flow[:, input_flow.shape[1]//2, input_flow.shape[2]//2]
            input_ego = np.array(ego_motion[start:start+data_length])
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
                count += 1


print(f'Total :: {count}')