import gzip
import cPickle
import numpy as np
from sklearn.decomposition import PCA

def pca(trainx, testx, p = 0.95, verbose=False, n_components=False, get_pca=False):
    pca = PCA(whiten=True)
    pca.fit(trainx)
    components, variance = 0, 0.0
    for components, ele in enumerate(pca.explained_variance_ratio_):
        variance += ele
        if variance > p: break
    components += 1
    if verbose:
        print 'n_components=%d'%components
    pca.set_params(n_components=components)
    pca.fit(trainx)

    trainx = pca.transform(trainx)
    testx = pca.transform(testx)

    ret = (trainx, testx)
    if n_components: ret += (components,)
    if get_pca: ret += (pca,)
    return ret

import theano
import theano.tensor as T
from theano.tensor.nlinalg import diag as __gdiag
__gx = T.vector('__gx', dtype='float32') 
__gxs = T.matrix('__gxs', dtype='float32') 
__gM = T.matrix('__gM', dtype='float32') 
__gv = __gx-__gxs
__gdist = __gdiag(__gv.dot(__gM).dot(__gv.T))
__dist1vsN = theano.function([__gx, __gxs, __gM], __gdist, allow_input_downcast=True)

__gv = (__gx-__gxs)/T.sqrt(__gx+__gxs+1e-20)
__gdist = __gdiag(__gv.dot(__gM).dot(__gv.T))
__dist1vsNchisquare = theano.function([__gx, __gxs, __gM], __gdist, allow_input_downcast=True)



def knn(train_x, test_x, train_y, test_y, M, K=5, verbose=False, cfmatrix=False):
    n = len(train_x)
    m = len(set(train_y))
    if M is None:
        M = np.eye(len(train_x[0]))
    acc = 0
    rec = np.zeros((m, m))
    for x, y in zip(test_x, test_y):
        dist = __dist1vsN(x, train_x, M) 

        predict = np.bincount(train_y[dist.argsort()[:K]]).argmax()
        if predict == y: acc += 1
        rec[y][predict] += 1
    if verbose:
        print '{}/{}'.format(acc, test_y.shape[0])

    acc = float(acc)/test_y.shape[0]*100 
    if cfmatrix:
        return acc, rec
    else:
        return acc

def knnchisqaure(train_x, test_x, train_y, test_y, M, K=5, verbose=False, cfmatrix=False):
    n = len(train_x)
    m = len(set(test_y))
    if M is None:
        M = np.eye(len(train_x[0]))
    acc = 0
    rec = np.zeros((m, m))
    for x, y in zip(test_x, test_y):
        dist = __dist1vsNchisqaure(x, train_x, M) 

        predict = np.bincount(train_y[dist.argsort()[:K]]).argmax()
        if predict == y: acc += 1
        rec[y][predict] += 1
    if verbose:
        print '{}/{}'.format(acc, test_y.shape[0])

    acc = float(acc)/test_y.shape[0]*100 
    if cfmatrix:
        return acc, rec
    else:
        return acc





def load_mnist(percentage=1.0, skip_valid=False):
    print 'Loading data ...'
    data = cPickle.load(gzip.open('/home/share/mnist/mnist.pkl.gz'))

    if percentage < 1.0:
        newdata =[]
        for (x, y) in data:
            l = x.shape[0]
            indices = np.random.choice(range(0, l), int(percentage * l))
            newdata.append((x[indices], y[indices]))
        data = newdata

    if skip_valid:
        data = (data[0], data[-1])

    print('{}: {}'.format('Mnist shape', '; '.join(map(lambda ele: 'x:{}, y:{}'.format(ele[0].shape, ele[1].shape), data))))
    return data

if __name__ == '__main__':
    train_data, test_data = load_mnist(percentage=0.1, skip_valid=True)
