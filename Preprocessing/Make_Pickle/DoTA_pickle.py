import os
import json
import numpy as np
import pickle as pkl
import logging
from skimage.transform import  resize
from math import ceil

TAG_FLOAT = 202021.25

image_resolution = (1280, 720)
flow_resolution = (960, 540)

flow_shape = (5,5,2)

THRESHOLD = 3
IOU_THRESHOLD = 0.4
PASS_SIZE = 3

base_path = 'E:/Detection-of-Traffic-Anomaly-master/dataset/'
save_base_path = 'D:/DoTA/result/'

STATE = 'val' # or 'val'

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = int(np.fromfile(f, np.int32, count=1))
    h = int(np.fromfile(f, np.int32, count=1))
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))	
    f.close()

    return flow

def load_flo(video_name, frame_id, bbox):
    file_path = f'{base_path}frames/{video_name}/flow/{str(frame_id).zfill(6)}.flo'
    image = read_flo(file_path)
    image = image[int(bbox[6]/image_resolution[1]*flow_resolution[1]):
                  int(bbox[7]/image_resolution[1]*flow_resolution[1]),
                  int(bbox[8]/image_resolution[0]*flow_resolution[0]):
                  int(bbox[9]/image_resolution[0]*flow_resolution[0])]
    return resize(image, flow_shape)

def make_bbox(bbox):
    temp = []
    temp.append(bbox[0]/image_resolution[0])
    temp.append(bbox[1]/image_resolution[1])
    temp.append(bbox[2]/image_resolution[0])
    temp.append(bbox[3]/image_resolution[1])
    return np.array(temp)

def interpolate(start, end, between, is_id = False, is_label = False):
    temp = np.linspace(start, end, between+1)[1:]
    result = []
    if is_label:
        for i in temp:
            result.append(ceil(i))
    elif is_id:
        for i in temp:
            result.append(int(i))
    else:
        for i in temp:
            result.append(i)    
    return result

def calculate_iou(yolo_bbox, JSON):
    result = 0
    yolo = [yolo_bbox[2], yolo_bbox[0], yolo_bbox[3], yolo_bbox[1]]
    for json_bbox in JSON:
        # x1, y1, x2, y2 순서로 변환
        
        json = [json_bbox['bbox'][0], json_bbox['bbox'][1], json_bbox['bbox'][2], json_bbox['bbox'][3]]

        intersection_x1 = max(yolo[0], json[0])
        intersection_y1 = max(yolo[1], json[1])
        intersection_x2 = min(yolo[2], json[2])
        intersection_y2 = min(yolo[3], json[3])

        intersection = max(0, intersection_x2-intersection_x1) * max(0, intersection_y2-intersection_y1)

        bbox1_area = abs((yolo[2]-yolo[0]) * (yolo[3]-yolo[1]))
        bbox2_area = abs((json[2]-json[0]) * (json[3]-json[1]))

        result = max(result, intersection / (bbox1_area + bbox2_area - intersection + 1e-7))

    return result

def ego_pickling(video_name, start, finish, length):
    """
    1. ego motion 데이터 불러오기
    2. 데이터 전체 피클링
    3. length만큼의 label을 0으로 처리
    4. label[start:finish] = 1
    5. 해당 dictionary pickling 
    """
    dictionary = {}
    try:
        with open(f'{base_path}frames/{video_name}/{video_name}.txt', 'r') as f:
            temp = f.read().strip()
        temp = temp.replace('\n',' ')
        temp = list(temp.split(' '))
        temp = list(map(float, temp))
        ego_motion = [] # ego motion 데이터 불러오기
        for index in range(0, len(temp), 3):
            ego_motion.append(temp[index:index+3])

        dictionary['ego_motion'] = ego_motion
        dictionary['label'] = [0] * length
        for index in range(start, finish):
            dictionary['label'][index] = 1
        
        print(f'Save Ego Pickle :: {video_name}')
        print(f'Save Path :: {save_base_path}{STATE}/for_ego/{video_name}.pkl')
        with open(file = f'{save_base_path}{STATE}/for_ego/{video_name}.pkl' , mode = 'wb') as f:
            pkl.dump(dictionary, f)
    except:
        logging.warning(f'FAIL TO MAKE PICKLE\t{video_name}')

def accident_pickling(video_name, start, finish, length):
    if not os.path.isfile(f'{base_path}frames/{video_name}/{video_name}.npy'): # BBox 정보가 없다면
        logging.warning(f'FAIL TO MAKE PICKLE(BBox)\t{video_name}')
    elif not os.path.isfile(f'{base_path}frames/{video_name}/{video_name}.txt'): # Ego Motion 정보가 없다면
        logging.warning(f'FAIL TO MAKE PICKLE(Ego)\t{video_name}')
    elif not os.path.isdir(f'{base_path}frames/{video_name}/flow'): # Flow 정보가 없다면
        logging.warning(f'FAIL TO MAKE PICKLE(Flow)\t{video_name}')
    else:
        with open(f'{base_path}frames/{video_name}/{video_name}.txt', 'r') as f:
            temp = f.read().strip()
        temp = temp.replace('\n',' ')
        temp = list(temp.split(' '))
        temp = list(map(float, temp))
        ego_motion = [] # ego motion 데이터 불러오기
        for index in range(0, len(temp), 3):
            ego_motion.append(temp[index:index+3])
        
        bbox = np.loadtxt(f'{base_path}frames/{video_name}/{video_name}.npy', delimiter=',')
        with open(f'{base_path}DoTA_annotations/annotations/{video_name}.json', 'r') as f:
            json_file = json.load(f) # json 파일
        
        now_object = []
        dictionary = {}

        for index in range(len(bbox)):
            if bbox[index][0] >= length:
                break
            # Optical Flow크기 5보다 큰지 확안
            if bbox[index][4]/image_resolution[1]*flow_resolution[1] < 5 or bbox[index][5]/image_resolution[1]*flow_resolution[1] < 5:
                logging.info(f'Too Small BBox\t{video_name} file Frame :: {index}')
            else: # Optical Flow가 크다 -> 이용 가능한 데이터
                # json 파일 확인해서 해당 bbox 겹치는지 확인하기
                if json_file['labels'][int(bbox[index][0])]['objects'] == []:
                    result_iou = 0
                else:
                    if IOU_THRESHOLD <= (calculate_iou(bbox[index][6:], json_file['labels'][int(bbox[index][0])]['objects'])):
                        result_iou = 1
                    else:
                        result_iou = 0
                ################## IOU 계산 함수 검증 완료
                if bbox[index][1] not in now_object: # 새로운 데이터라면
                    dictionary[int(bbox[index][1])] = {'bbox' : [make_bbox(bbox[index][2:6])], # numpy array
                                                'frame_id' : [int(bbox[index][0])], # int
                                                'ego_motion' : [ego_motion[int(bbox[index][0])-1]],  # list
                                                'flow' : [load_flo(video_name, int(bbox[index][0]), bbox[index])], # numpy array
                                                'label' : [result_iou]} 
                    now_object.append(bbox[index][1])
                else: # 이전에 발견된 데이터라면
                    if int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1] <= THRESHOLD: # 감당가능한 범위 내라면
                        #print(f'Frame Number :: {int(bbox[index][0])}')
                        dictionary[int(bbox[index][1])]['bbox'] += interpolate(dictionary[int(bbox[index][1])]['bbox'][-1], make_bbox(bbox[index][2:6]), int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1])
                        dictionary[int(bbox[index][1])]['ego_motion'] += interpolate(dictionary[int(bbox[index][1])]['ego_motion'][-1], ego_motion[int(bbox[index][0])-1], int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1])
                        dictionary[int(bbox[index][1])]['flow'] += interpolate(dictionary[int(bbox[index][1])]['flow'][-1], load_flo(video_name, int(bbox[index][0]), bbox[index]), int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1])
                        dictionary[int(bbox[index][1])]['label'] += interpolate(dictionary[int(bbox[index][1])]['label'][-1], result_iou, int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1], is_label = True)
                        dictionary[int(bbox[index][1])]['frame_id'] += interpolate(dictionary[int(bbox[index][1])]['frame_id'][-1], int(bbox[index][0]), int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1], is_id=True)
                        
                    else: # 범위 밖의 데이터
                        start_frame = dictionary[bbox[index][1]]['frame_id'][0]

                        if len(dictionary[bbox[index][1]]['frame_id']) >= PASS_SIZE:
                            print(f'Save File {save_base_path}{STATE}/for_train/{video_name}_{int(bbox[index][1])}_{start_frame}.pkl')
                            with open(file = f'{save_base_path}{STATE}/for_train/{video_name}_{int(bbox[index][1])}_{start_frame}.pkl', mode = 'wb') as f:
                                pkl.dump(dictionary[int(bbox[index][1])], f)
                        #print(dictionary[bbox[index][1]]['frame_id'])
                        #print(bbox[index][1])
                        #print(f'Frame Number :: {int(bbox[index][0])}')
                        dictionary[int(bbox[index][1])] = {}

                        temp = {'bbox' : [make_bbox(bbox[index][2:6])], # numpy array
                                                'frame_id' : [int(bbox[index][0])], # int
                                                'ego_motion' : [ego_motion[int(bbox[index][0])-1]],  # list
                                                'flow' : [load_flo(video_name, int(bbox[index][0]), bbox[index])], # numpy array
                                                'label' : [result_iou]} 
                    
                        dictionary[int(bbox[index][1])] = temp

        for key in dictionary.keys():
            start_frame = dictionary[key]['frame_id'][0]
            if len(dictionary[key]['frame_id']) >=PASS_SIZE:
                print(f'Save File {save_base_path}{STATE}/for_train/{video_name}_{key}_{start_frame}.pkl')
                with open(file = f'{save_base_path}{STATE}/for_train/{video_name}_{key}_{start_frame}.pkl', mode = 'wb') as f:
                    pkl.dump(dictionary[key], f)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = f'{save_base_path}log.txt'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

with open(f'{base_path}{STATE}_split.txt' , 'r') as f:
    splited_data = f.read().strip().split('\n') # 각 State에 맞게끔 분리되어있는 데이터의 이름

temp = os.listdir(f'{base_path}frames/')

possible_video = [instance for instance in temp if instance in splited_data] # State에 맞으며 실제로도 존재하는 영상의 이름
# D:/Detection-of-Traffic-Anomaly-master/dataset/frames/0qfbmt4G8Rw_000306

with open(f'{base_path}metadata_{STATE}.json', 'r') as f:
    META = json.load(f) # 각 state의 메타데이터

for video_name in possible_video:
    #video_name = 'test'
    meta_data = META[video_name] # 해당 영상의 메타데이터 불러오기
    if meta_data['anomaly_class'].split(':')[0] == 'ego': # 자기 혼자 발생하는 사고
        ego_pickling(video_name = video_name, start = meta_data['anomaly_start'], finish = meta_data['anomaly_end'], length = meta_data['num_frames'])
    else: # 다른 차량의 사고인 경우
        accident_pickling(video_name = video_name, start = meta_data['anomaly_start'], finish = meta_data['anomaly_end'], length = meta_data['num_frames'])