data_root = '/workspace/RECURRENCE_DATA/'
data_length = 16
stride = 1

data_shuffle = True
data_batch = 256
num_workers = 0
split_ratio = [0.8, 0.2] # train, validation data ratio

flow_eps = [0.01, 0.009]
bbox_eps = [0.0003, 0.0005, 0.0005, 0.0006]
ego_eps = [0.01, 0.01, 0.01]

