import numpy as np
import matplotlib.pyplot as plt
import sys
#file = np.loadtxt('./result/image_left_ego.txt')
color=['r','g','b','y']
label=['reconst','GT','path 2']
for i in range(1, len(sys.argv)):
    
    path_vo = np.loadtxt(sys.argv[i])
    if i ==2:
        path_vo[:,0] = np.arctan2(path_vo[:,4], path_vo[:,0])
    
    plt.plot(path_vo[:,0],color[i-1],label=label[i-1])
    
plt.xlabel('frame')
plt.ylabel('yaw')
plt.title('PATH')
plt.legend()
plt.show()