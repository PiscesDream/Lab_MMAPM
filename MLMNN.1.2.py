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
        dim = trainx.shape[1]
        print 'train-acc: %.3f%% test-acc: %.3f%% %s'%(knn(trainx, trainx, y, y, np.eye(dim), self.K), knn(trainx, testx, y, testy, np.eye(dim), self.K), ' '*30)


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

        newx[np.isnan(newx)] = 0
        newx[np.isinf(newx)] = 0
        return newx

    def dist(self, xi, xj):
        return self.dist2(xi, xj)
    def dist2(self, xi, xj): # same as dist1
        acc = 0.0
        for i in xrange(self.M):
            xi_ = self.PCAs[i].transform(xi.reshape(1, self.M, -1)[:, i, :])
            xj_ = self.PCAs[i].transform(xj.reshape(1, self.M, -1)[:, i, :])
            acc += (xi_-xj_).dot(self.Ms[i]._M).dot((xi_-xj_).T)
        return acc
    def dist1(self, xi, xj):
        xi, xj = self.transform(np.array([xi, xj]))
        return ((xi-xj)**2).sum()



if __name__ == '__main__':
    K = 100 
    T = 1.0
    M = 10

    trainx, testx, trainy, testy = cPickle.load(open('./features/[K={}][T={}]BoWInGroup.pkl'.format(K, T),'r'))
    print trainx.shape
    print trainy.shape
    
    mlmnn = MLMNN(M=10, K=7, mu=0.5, Ppca=0.995,
                maxS=10000, lr=8e-5, max_iter=100, reset=20)
    mlmnn.fit(trainx, trainy, testx, testy, verbose=True)

