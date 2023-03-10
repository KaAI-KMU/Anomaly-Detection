import numpy as np
import os
from skimage.transform import  resize
import pickle as pkl

root = 'data/'

video_names = [name.split('.')[0] for name in os.listdir(root) if name.endswith('.txt')]
TAG_FLOAT = 202021.25

image_resolution = (1280, 720)
flow_resolution = (960, 540)

flow_shape = (5,5,2)

THRESHOLD = 3

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
    file_path = f'data/{video_name}/{str(frame_id+1).zfill(6)}.flo'
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

def interpolate(start, end, between, is_id = False):
    temp = np.linspace(start, end, between+1)[1:]
    result = []
    for i in temp:
        if is_id:
            i = int(i)
        result.append(i)
    return result

for video_name in video_names:
    save_dir = f'result/{video_name}/'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(f'{root}{video_name}.txt', 'r') as f:
        temp = f.read().strip()
    temp = temp.replace('\n',' ')
    temp = list(temp.split(' '))
    temp = list(map(float, temp))
        #temp = list(map(float, f.read().replace('\n',' ').split(' ')))
    ego_motion = []
    for index in range(0, len(temp), 3):
        ego_motion.append(temp[index:index+3])
    bbox = np.loadtxt(f'{root}{video_name}.npy', delimiter=',')
    # frame, id, center_x, center_y, width, height, top, top+height, left, left+width
    ### BBox 파일(numpy) Ego 파일(txt) 불러오기 완료

    now_object = []
    dictionary = {}

    for index in range(len(bbox)-1):
        if bbox[index][0] >= len(ego_motion)-1:
            break
        if bbox[index][1] not in now_object: # 신규 id
            
            #print(f'Frame Number :: {int(bbox[index][0])}')

            dictionary[int(bbox[index][1])] = {'bbox' : [make_bbox(bbox[index][2:6])], # numpy array
                                            'frame_id' : [int(bbox[index][0])], # int
                                            'ego_motion' : [ego_motion[int(bbox[index][0])-1]],  # list
                                            'flow' : [load_flo(video_name, int(bbox[index][0]), bbox[index])]} # numpy array
            now_object.append(bbox[index][1])
        else: # 이전에 있던 데이터라면
            if int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1] <= THRESHOLD: # 감당가능한 범위 내라면
                #print(f'Frame Number :: {int(bbox[index][0])}')
                dictionary[int(bbox[index][1])]['bbox'] += interpolate(dictionary[int(bbox[index][1])]['bbox'][-1], make_bbox(bbox[index][2:6]), int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1])
                dictionary[int(bbox[index][1])]['ego_motion'] += interpolate(dictionary[int(bbox[index][1])]['ego_motion'][-1], ego_motion[int(bbox[index][0])-1], int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1])
                dictionary[int(bbox[index][1])]['flow'] += interpolate(dictionary[int(bbox[index][1])]['flow'][-1], load_flo(video_name, int(bbox[index][0]), bbox[index]), int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1])
                dictionary[int(bbox[index][1])]['frame_id'] += interpolate(dictionary[int(bbox[index][1])]['frame_id'][-1], int(bbox[index][0]), int(bbox[index][0]) - dictionary[bbox[index][1]]['frame_id'][-1], is_id=True)
            else: # 범위 밖의 데이터
                start_frame = dictionary[bbox[index][1]]['frame_id'][0]
                print(f'Save File {save_dir}{video_name}_{int(bbox[index][1])}_{start_frame}.pkl')
                with open(file = f'{save_dir}{video_name}_{int(bbox[index][1])}_{start_frame}.pkl', mode = 'wb') as f:
                    pkl.dump(dictionary[bbox[index][1]], f)
                #print(dictionary[bbox[index][1]]['frame_id'])
                #print(bbox[index][1])
                #print(f'Frame Number :: {int(bbox[index][0])}')
                dictionary[int(bbox[index][1])] = {}
                temp = {'bbox' : [make_bbox(bbox[index][2:6])], # numpy array
                                            'frame_id' : [int(bbox[index][0])], # int
                                            'ego_motion' : [ego_motion[int(bbox[index][0])-1]],  # list
                                            'flow' : [load_flo(video_name, int(bbox[index][0]), bbox[index])]} # numpy array
            
                dictionary[int(bbox[index][1])] = temp
    for key in dictionary.keys():
        start_frame = dictionary[key]['frame_id'][0]
        print(f'Save File {save_dir}{video_name}_{key}_{start_frame}.pkl')
        with open(file = f'{save_dir}{video_name}_{key}_{start_frame}.pkl', mode = 'wb') as f:
            pkl.dump(dictionary[bbox[index][1]], f)
