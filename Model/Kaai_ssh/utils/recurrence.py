import numpy as np

def rec_plot_module(s, eps=0.01, steps = 255):
    N = s.size
    S = np.repeat(s[None,:], N, axis = 0)
    Z = np.floor(np.abs(S-S.T)/eps)
    Z[Z>steps] = steps
    
    return Z

def rec_plot(s):
    storage = np.zeros((s.shape[1], s.shape[0], s.shape[0]))
    for index in range(s.shape[1]):
        storage[index, :, :] = rec_plot_module(s[:, index])
    return storage