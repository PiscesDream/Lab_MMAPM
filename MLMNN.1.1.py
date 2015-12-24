# Multi-stage LMNN
import numpy as np
import numpy.linalg as linalg
import cPickle
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric  
from sklearn.cross_validation import train_test_split 

from metric_learn import LMNN as _LMNN
from Metric.LMNN import LMNN
from Metric.LMNN2 import LMNN as LMNN_GPU
from Metric.common import knn, pca

class MLMNN(object):
    def __init__(self, M, K, mu, maxS, lr=1e-7, max_iter=100, reset=20, Ppca=0.95):
        self.M = M
        self.K = K
        self.mu = mu
        self.maxS = maxS
        self.lr = lr
        self.max_iter = max_iter
        self.reset = reset
        self.Ppca = Ppca 

    def fit(self, x, y, testx, testy, verbose=False, autosave=True):
        self.x = x = x.reshape(x.shape[0], self.M, -1)
        self.y = y
        self.testx = testx = testx.reshape(testx.shape[0], self.M, -1)
        self.testy = y

        self.Ms = []
        self.PCAs = []

        for i in xrange(self.M):
            trainx, testx, _pca = pca(self.x[:, i, :], self.testx[:, i, :], self.Ppca, verbose=verbose, get_pca=True)
            lmnn = LMNN_GPU(K=self.K, mu=self.mu, maxS=self.maxS, dim=trainx.shape[1])

            print '[%d]pre-train-acc: %.3f%% pre-test-acc: %.3f%% %s'%(i, knn(trainx, trainx, y, y, lmnn._M, self.K), knn(trainx, testx, y, testy, lmnn._M, self.K), ' '*30)
            lmnn.fit(trainx, y, lr=self.lr, max_iter=self.max_iter, reset=self.reset, verbose=False)
            print '[%d]post-train-acc: %.3f%% post-test-acc: %.3f%% %s'%(i, knn(trainx, trainx, y, y, lmnn._M, self.K), knn(trainx, testx, y, testy, lmnn._M, self.K), ' '*30)


            self.Ms.append(lmnn)
            self.PCAs.append(_pca)

        trainx = self.transform(self.x)
        testx = self.transform(self.testx)

        trainx[np.isnan(trainx)] = 0
        trainx[np.isinf(trainx)] = 0

        testx[np.isnan(testx)] = 0
        testx[np.isinf(testx)] = 0
        print 'Final train shape:',trainx.shape
        print 'Final test shape:',testx.shape
        trainx, testx = pca(trainx, testx, self.Ppca, verbose=True)
        print 'Final train shape:',trainx.shape
        print 'Final test shape:',testx.shape
    
        # integrated metric
        lmnn = LMNN_GPU(K=self.K, mu=self.mu, maxS=10000, dim=trainx.shape[1])
        for i in xrange(10):
            print '[%d]pre-train-acc: %.3f%% pre-test-acc: %.3f%% %s'%(i, knn(trainx, trainx, y, y, lmnn._M, self.K), knn(trainx, testx, y, testy, lmnn._M, self.K), ' '*30)
            lmnn.fit(trainx, y, lr=2e-5, max_iter=50, reset=50, verbose=False)
            print '[%d]post-train-acc: %.3f%% post-test-acc: %.3f%% %s'%(i, knn(trainx, trainx, y, y, lmnn._M, self.K), knn(trainx, testx, y, testy, lmnn._M, self.K), ' '*30)

        if autosave:
            print 'Auto saving ...'
            cPickle.dump(self, open('temp.MLMNN', 'w'))
        return self


    def transform(self, x):
        x = x.reshape(x.shape[0], self.M, -1)
        newx = []
        for i in xrange(self.M):
            newx.append(self.PCAs[i].transform(x[:, i, :]).dot(self.Ms[i].L))
            print 'MS[{}] Size: {} Shape: {} segment: {}'.format(i, self.Ms[i]._M.shape, linalg.matrix_rank(self.Ms[i]._M), newx[-1].shape)
        newx = np.hstack(newx)
        return newx


if __name__ == '__main__':
    K = 100 
    T = 1.0
    M = 10

    trainx, testx, trainy, testy = cPickle.load(open('./features/[K={}][T={}]BoWInGroup.pkl'.format(K, T),'r'))
    print trainx.shape
    print trainy.shape
    
    mlmnn = MLMNN(M=4, K=5, mu=0.5, Ppca=0.995,
                maxS=10000, lr=5e-5, max_iter=200, reset=50)
    mlmnn.fit(trainx, trainy, testx, testy, verbose=True)

