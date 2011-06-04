# the kernel for xx

import numpy as np

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def zero_kernel(x, y):
#    return 0
    ans = np.zeros((x.shape[0], y.shape[0]))
    #print ans.shape
    return ans

def multichannel_wrapper(C, _K):
    def K(_x, _y):
        _x = _x.reshape(_x.shape[0], C, -1)
        _y = _y.reshape(_y.shape[0], C, -1)
        ans = np.zeros((_x.shape[0], _y.shape[0]))
        #for c in [1]: # 85.93 
        #for c in [0]: # 33.46 
        for c in range(C): # 88.53
            #x = _x[:, c, :] 
            #y = _y[:, c, :] 
            ans += _K(_x[:, c, :], _y[:, c, :])
        ans = np.exp(-ans)
        return ans;
    return K

def histogram_intersection_kernel(x, y):
    #normalize: a / np.array( a.sum(2).reshape(a.sum(2).shape+(1,)), dtype=float)
    #x = x / x.sum(1).reshape(x.shape[0],1)

    x = np.tile(x, (y.shape[0],1,1)).swapaxes(0,1)
    y = np.tile(y, (x.shape[0],1,1))
    #esp = 1e-20

    d = 1.0 - np.minimum(x, y)/(np.maximum(x, y))#esp)
    d[np.isnan(d)] = 0
    d = d.sum(2)
    d /= d.max()
    # print d.max()

    return d

def chi_square_kernel(x, y):
    x = np.tile(x, (y.shape[0],1,1)).swapaxes(0,1)
    y = np.tile(y, (x.shape[0],1,1))

    #    print (x+y).min() == 0
    #esp = 1e-20
    d = 0.5*((x-y)**2)/(x+y)#+esp)
    d[np.isnan(d)] = 0
    d = d.sum(2)
    d /= d.max() 

    # alpha == 1000, best = 73%
    #alpha = 10000
    #d = (0.5*((x-y)**2)/(x+y+alpha)).sum(2)
    #print d.max()

    return d


    


