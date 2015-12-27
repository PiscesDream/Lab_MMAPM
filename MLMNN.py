# Multi-stage LMNN
import matplotlib
matplotlib.use('Agg')
from tSNE.draw import visualize as visualize

import numpy as np
import numpy.linalg as linalg
import cPickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from metric_learn import LMNN as _LMNN
from Metric.LMNN import LMNN
#from Metric.LMNN2 import LMNN as LMNN_GPU
from Metric.common import knn

import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg
import numpy.linalg as nplinalg
from Metric.LDL import LDL
import sys

class MLMNN(object):
    def __init__(self, M, K, mu, dim, lmbd, 
                localM=True, globalM=True, normalize_axis=None, kernelf=None): 
        sys.setrecursionlimit(10000)
        self.localM = localM
        self.globalM = globalM 
        
        self.M = M
        self.K = K
        self.mu = mu
        self.dim = dim # dim = localdim = globladim = pcadim 
        self.lmbd = lmbd # lmbd*local + (1-lmbd)*global
        self.normalize_axis = normalize_axis
        self.kernelf = kernelf

        self._x = T.tensor3('_x', dtype='float32')
        self._stackx = T.tensor3('_stackx', dtype='float32')
        self._y = T.ivector('_y') 
        self._lr = T.vector('_lr', dtype='float32')
        self._set = T.imatrix('_set')
        self._neighborpairs = T.imatrix('_neighborpairs')

    def build(self):
        self.debug = []
        lM = []
        lpullerror = []
        lpusherror = []
        lupdate = []
        for i in xrange(self.M):
            if not self.localM: 
                lM.append(theano.shared(value=np.eye(self.dim, dtype='float32'), name='M', borrow=True))
                lpullerror.append(0.0)
                lpusherror.append(0.0)
                continue
            M = theano.shared(value=np.eye(self.dim, dtype='float32'), name='M', borrow=True)
            pullerror, pusherror = self._local_error(M, i)
            error = (1-self.mu) * pullerror + self.mu * pusherror
            update = (M, M - self._lr[i] * T.grad(error, M))

            lM.append(M)
            lpullerror.append((1-self.mu)*pullerror)
            lpusherror.append(self.mu*pusherror)
            lupdate.append(update)

        self.lM = lM
        self.lpusherror = lpusherror
        self.lpullerror = lpullerror
        self.lupdate = lupdate

        #gError = 0.0
        gM = []
        gpullerror = []
        gpusherror = []
        gupdate = []
        for i in xrange(self.M):
            if not self.globalM: 
                gM.append(theano.shared(value=np.eye(self.dim, dtype='float32'), name='M', borrow=True))
                gpullerror.append(0.0)
                gpusherror.append(0.0)
                continue
            M = theano.shared(value=np.eye(self.dim, dtype='float32'), name='M', borrow=True)
            if i == 0:
                pullerror, pusherror = self._global_error(M, i, None)
            else:
                pullerror, pusherror = self._global_error(M, i, gM[-1])
            error = (1-self.mu) * pullerror + self.mu * pusherror
        #    gError += error#*(float(i+1)/self.M)
            update = (M, M - self._lr[i+self.M] * T.grad(error, M))

            gM.append(M)
            gpullerror.append((1-self.mu)*pullerror)
            gpusherror.append(self.mu*pusherror)
            gupdate.append(update)
#       if self.globalM: 
#           gupdate = [(gM[i], gM[i] - self._lr[i+self.M]*T.grad(gError, M)) for i in xrange(self.M)]

        self.gM = gM
        self.gpusherror = gpusherror
        self.gpullerror = gpullerror
        self.gupdate = gupdate


    def fit(self, x, y, testx, testy,
        maxS=100, lr=1e-7, max_iter=100, reset=20, 
        Part=None,
        verbose=False, autosave=True): #


        if Part is not None and Part != self.M:
            x = x.reshape(x.shape[0], self.M, -1)[:, :Part, :].reshape(x.shape[0], -1)
            testx = testx.reshape(testx.shape[0], self.M, -1)[:, :Part, :].reshape(testx.shape[0], -1)
            self.M = Part

        # normalize in each part
        #x = normalize(x.reshape(x.shape[0]*self.M, -1), 'l1').reshape(x.shape[0], -1)
        #testx = normalize(testx.reshape(testx.shape[0]*self.M, -1), 'l1').reshape(testx.shape[0], -1)
        # normalize in each sample
        if self.kernelf:
            x = self.kernelf(x)
            testx = self.kernelf(testx)
        if self.normalize_axis:
            x = normalize(x, axis=self.normalize_axis)
            testx = normalize(testx, axis=self.normalize_axis)

        self.maxS = maxS
        orix = x
         
        if self.dim == -1: # no pca needed
            self.dim = x.shape[1]
            self.didpca = False 
        else: # do the pca and store the solver
            self.didpca = True
            x = x.reshape(x.shape[0] * self.M, -1)
            print 'Splited x.shape:', x.shape
            pca = PCA(n_components=self.dim)
            x = pca.fit_transform(x)
            self.pca = pca
        print x.shape
        x = x.reshape(-1, self.M, self.dim)
        print 'Final x.shape:', x.shape
        stackx = x.copy()
        for i in xrange(1, self.M):
            stackx[i] += stackx[i-1]

        try:
            lM = self.lM
            gM = self.gM
        except:
            self.build()
            lM = self.lM
            gM = self.gM

        updates = []  
        givens = {self._x: np.asarray(x, dtype='float32'),
                  self._y: np.asarray(y, dtype='int32')}
        if self.localM: updates.extend(self.lupdate)
        if self.globalM: 
            updates.extend(self.gupdate)
            givens.update({self._stackx: np.asarray(stackx, dtype='float32')})

        self.train_local_model = theano.function(
            [self._set, self._neighborpairs, self._lr], 
            [
            T.stack(self.lpullerror), 
            T.stack(self.gpullerror), 
            T.stack(self.lpusherror),
            T.stack(self.gpusherror)],
            updates = updates,
            givens = givens)

#       get_debug = theano.function(
#           [self._set], 
#           self.debug,
#           givens = {self._stackx: np.asarray(stackx, dtype='float32')})

        __x = x
        lr = np.array([lr]*(self.M*2), dtype='float32')
        for _ in xrange(40):
            neighbors = self._get_neighbors(__x, y)
            t = 0
            while t < max_iter:
                if t % reset == 0:
                    active_set = self._get_active_set(x, y, neighbors)
                    last_error = np.array([np.inf]*(self.M*2))

                print 'Iter: {} lr: {} '.format(t, lr)

                res = np.array(self.train_local_model(active_set, neighbors, lr)).reshape(-1, self.M*2)
                error = res.T.sum(1)
                print '\tlpull:{}\tgpull:{}\n\tlpush:{}\tgpush:{}'.\
                    format(res[0, :self.M], res[0, self.M:],\
                           res[1, :self.M], res[1, self.M:])
                
                #print np.array(get_debug(active_set))

                for i in xrange(self.M):
                    _M = lM[i].get_value() 
                    lM[i].set_value(np.array(self._numpy_project_sd(_M), dtype='float32'))
                for i in xrange(self.M):
                    _M = gM[i].get_value() 
                    gM[i].set_value(np.array(self._numpy_project_sd(_M), dtype='float32'))

                lr = lr*1.01*(last_error>error) + lr*0.5*(last_error<=error) 

                last_error = error
                t += 1
            

            __x = self.transform(orix)
            __testx = self.transform(testx)
            train_acc, train_cfm = knn(__x, __x, y, y, None, self.K, cfmatrix=True)

            test_acc, test_cfm = knn(__x, __testx, y, testy, None, self.K, cfmatrix=True)
            print 'shape: {}'.format(__x.shape)
            print 'train-acc: %.3f%% test-acc: %.3f%% %s'%(\
                train_acc, test_acc, ' '*30)
            print 'train confusion matrix:\n {}\ntest confusion matrix:\n {}'.format(
                train_cfm, test_cfm)

#           __y = label_binarize(y, classes=range(8))
#           __testy = label_binarize(testy, classes=range(8))
#           svm = OneVsRestClassifier(SVC(kernel='rbf')).fit(__x, __y)
#           train_acc = float((svm.predict(__x) == __y).sum())/y.shape[0]
#           test_acc = float((svm.predict(__testx) == __testy).sum())/testy.shape[0]
#           print '[svm]train-acc: %.3f%% test-acc: %.3f%% %s'%(\
#               train_acc, test_acc, ' '*30)

#           print 'visualizing round{} ...'.format(_)
#           title = 'round{}.train'.format(_) 
#           visualize(__x, y, title+'_acc{}'.format(train_acc), 
#               './visualized/{}.png'.format(title))
#           title = 'round{}.test'.format(_) 
#           visualize(__testx, testy, title+'_acc{}'.format(test_acc),
#               './visualized/{}.png'.format(title))
            
        if autosave:
            print 'Auto saving ...'
            self.save('temp.MLMNN')
        return self

    def save(self, filename):
        cPickle.dump((self.__class__, self.__dict__), open(filename, 'w'))

    @staticmethod
    def load(filename):
        cls, attributes = cPickle.load(open(filename, 'r'))
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

    def transform(self, x):
        #x = normalize(x.reshape(x.shape[0]*self.M, -1)).reshape(x.shape[0], -1)
        if self.kernelf:
            x = self.kernelf(x)
        if self.normalize_axis:
            x = normalize(x, axis=self.normalize_axis)

        n = x.shape[0]
        x = x.reshape(n*self.M, -1)
        if self.didpca: x = self.pca.transform(x)
        x = x.reshape(n, self.M, self.dim)
        stackx = np.zeros((n,self.dim))
        newx = []
        localdim = globaldim = 0
        for i in xrange(self.M):
            if self.localM:
                L = LDL(self.lM[i].get_value(), combined=True)
                newx.append(x[:, i, :].dot(L)*self.lmbd )
            #print 'MS[{}] Size: {} rank: {} segment: {}'.format(i, self.lM[i].get_value().shape, nplinalg.matrix_rank(self.lM[i].get_value()), newx[-1].shape)

            if self.globalM:
                stackx += x[:, i, :]
        newx.append(stackx.dot(LDL(self.gM[-1].get_value(), combined=True))*(1-self.lmbd))
        newx = np.hstack(newx)
        newx[np.isnan(newx)] = 0
        newx[np.isinf(newx)] = 0
        return newx

    def _numpy_project_sd(self, mat):
        eig, eigv = nplinalg.eig(mat)
        eig[eig < 0] = 0
        eig = np.diag(eig)
        return eigv.dot(eig).dot(nplinalg.inv(eigv))

    def _global_error(self, targetM, i, lastM):
        mask = T.neq(self._y[self._set[:, 1]], self._y[self._set[:, 2]])
        f = T.tanh #T.nnet.sigmoid
        if i == 0:
            # pull_error for global 0
            pull_error = 0.
            ivectors = self._stackx[:, i, :][self._neighborpairs[:, 0]]
            jvectors = self._stackx[:, i, :][self._neighborpairs[:, 1]]
            diffv = ivectors - jvectors
            pull_error = linalg.trace(diffv.dot(targetM).dot(diffv.T))

            # push_error for global 0
            push_error = 0.0
            ivectors = self._stackx[:, i, :][self._set[:, 0]]
            jvectors = self._stackx[:, i, :][self._set[:, 1]]
            lvectors = self._stackx[:, i, :][self._set[:, 2]]
            diffij = ivectors - jvectors
            diffil = ivectors - lvectors
            lossij = diffij.dot(targetM).dot(diffij.T)
            lossil = diffil.dot(targetM).dot(diffil.T)
            #cur_prediction = T.diag(lossij - lossil)
            cur_prediction = f(T.diag(lossil - lossij))

            ivectors = self._stackx[:, i-1, :][self._set[:, 0]]
            jvectors = self._stackx[:, i-1, :][self._set[:, 1]]
            lvectors = self._stackx[:, i-1, :][self._set[:, 2]]
            diffij = ivectors - jvectors
            diffil = ivectors - lvectors
            lossij = diffij.dot(diffij.T)
            lossil = diffil.dot(diffil.T)
            #lst_prediction = T.diag(lossij - lossil)
            lst_prediction = f(T.diag(lossil - lossij))
            push_error = T.sum(mask*(lst_prediction - cur_prediction))

        else:
            ivectors = self._stackx[:, i, :][self._neighborpairs[:, 0]]
            jvectors = self._stackx[:, i, :][self._neighborpairs[:, 1]]
            diffv1 = ivectors - jvectors
            distMcur = diffv1.dot(targetM).dot(diffv1.T)
            ivectors = self._stackx[:, i-1, :][self._neighborpairs[:, 0]]
            jvectors = self._stackx[:, i-1, :][self._neighborpairs[:, 1]]
            diffv2 = ivectors - jvectors
            distMlast = diffv2.dot(lastM).dot(diffv2.T)
            pull_error = linalg.trace(T.maximum(distMcur - distMlast + 1, 0))


            # self.debug.append( self._y[self._set[:, 0] )

            push_error = 0.0
            ivectors = self._stackx[:, i, :][self._set[:, 0]]
            jvectors = self._stackx[:, i, :][self._set[:, 1]]
            lvectors = self._stackx[:, i, :][self._set[:, 2]]
            diffij = ivectors - jvectors
            diffil = ivectors - lvectors
            lossij = diffij.dot(targetM).dot(diffij.T)
            lossil = diffil.dot(targetM).dot(diffil.T)
            #cur_prediction = T.diag(lossij - lossil)
            cur_prediction = f(T.diag(lossil - lossij))

            ivectors = self._stackx[:, i-1, :][self._set[:, 0]]
            jvectors = self._stackx[:, i-1, :][self._set[:, 1]]
            lvectors = self._stackx[:, i-1, :][self._set[:, 2]]
            diffij = ivectors - jvectors
            diffil = ivectors - lvectors
            lossij = diffij.dot(lastM).dot(diffij.T)
            lossil = diffil.dot(lastM).dot(diffil.T)
            #lst_prediction = T.diag(lossij - lossil)
            lst_prediction = f(T.diag(lossil - lossij))
            push_error = T.sum(mask*(lst_prediction - cur_prediction))

        return pull_error, push_error 

    def _local_error(self, targetM, i):
        pull_error = 0.
        ivectors = self._x[:, i, :][self._neighborpairs[:, 0]]
        jvectors = self._x[:, i, :][self._neighborpairs[:, 1]]
        diffv = ivectors - jvectors
        pull_error = linalg.trace(diffv.dot(targetM).dot(diffv.T))

        push_error = 0.0
        ivectors = self._x[:, i, :][self._set[:, 0]]
        jvectors = self._x[:, i, :][self._set[:, 1]]
        lvectors = self._x[:, i, :][self._set[:, 2]]
        diffij = ivectors - jvectors
        diffil = ivectors - lvectors
        lossij = diffij.dot(targetM).dot(diffij.T)
        lossil = diffil.dot(targetM).dot(diffil.T)
        mask = T.neq(self._y[self._set[:, 0]], self._y[self._set[:, 2]])
        push_error = linalg.trace(mask*T.maximum(lossij - lossil + 1, 0))

#       print np.sqrt((i+1.0)/self.M)
#       pull_error = pull_error * np.sqrt((i+1.0)/self.M)
#       push_error = push_error * np.sqrt((i+1.0)/self.M)

        return pull_error, push_error 

    def _get_active_set(self, x, y, neighbors):
        result = []
        ijcandidates = neighbors[np.random.choice(range(neighbors.shape[0]), size=(self.maxS))]
        lcandidates = np.random.randint(0, x.shape[0], size=(self.maxS, 1) )
        ijl = np.hstack([ijcandidates, lcandidates])
        return np.array(ijl, dtype='int32')
#       for i, j, l in ijl: 
#           if y[i] == y[l]: continue
#           result.append((i, j, l)) #vij, vil ))
#       return np.array(result, dtype='int32')

    def _get_neighbors(self, x, y):
        # shared neighbor
        n = x.shape[0]
        x = x.reshape(n, -1)
        neighbors = np.zeros((n, self.K), dtype=int)
        yset = np.unique(y)
        COST = 0.0
        for c in yset:
            mask = y==c
            ind = np.arange(n)[mask]
            _x = x[mask]

            for i in xrange(n):
                if y[i] != c: continue
                v = x[i] - _x
                cost = (v**2).sum(1)#np.diag(v.dot(self._M).dot(v.T))
                neighbors[i] = ind[cost.argsort()[1:self.K+1]]
                COST += sum(sorted(cost)[1:self.K+1])
        print 'neighbor cost: {}'.format(COST)
        #self.neighbors = neighbors

        # repeat an arange array, stack, and transpose
        #self.neighborpairs = np.array(neighborpairs, dtype='int32')
        return np.array(np.vstack([np.repeat(np.arange(x.shape[0]), self.K), neighbors.flatten()]).T, dtype='int32')

def square_kernel(x):
    #x = np.hstack([x, x**2])
    return x

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf, precision=3)

    K = 200 
    Time = 0.1

    trainx, testx, trainy, testy = cPickle.load(open('./features/[K={}][T={}]BoWInGroup.pkl'.format(K, Time),'r'))
    print trainx.shape
    print trainy.shape
   
    mlmnn = MLMNN(M=1, K=5, mu=0.5, dim=100, lmbd=0.5, 
                    normalize_axis=0,
                    kernelf=None,
                    localM=True, globalM=True) 
    mlmnn.fit(trainx, trainy, testx, testy, \
        maxS=5000, lr=2, max_iter=50, reset=10,
        Part=None,
        verbose=True)


