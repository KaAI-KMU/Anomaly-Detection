# 배치크기
batch = 32

# 인코더가 공통으로 만들 feature의 차원
feature_space = 512

# feature map 비중
alpha = 0.4 # bbox
beta = 0.4 # flow
gamma = 0.2 # ego

# 데이터 time sequence
sequence = 16