import numpy as np

def rec_plot_module(s, eps=0.01, steps = 255):
    N = s.size
    S = np.repeat(s[None,:], N, axis = 0)
    Z = np.floor(np.abs(S-S.T)/eps)
    Z[Z>steps] = steps
    
    return Z

def rec_plot(s, epses, flow = False):
    s = np.array(s)
    storage = np.zeros((s.shape[-1], s.shape[0], s.shape[0]))
    if not flow:
        for index in range(s.shape[-1]):
            storage[index, :, :] = rec_plot_module(s[:, index], eps = epses[index])
    elif flow:
        for index in range(s.shape[-1]):
            storage[index, :, :] = rec_plot_module(s[:, s.shape[1]//2, s.shape[2]//2, index], eps = epses[index])
    return storage