from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import cPickle
import numpy as np
from names import *



import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg
import numpy as np
import cPickle

from Metric.common import knn
from sklearn.decomposition import PCA

import sys
sys.path.append('/home/shaofan/Projects') 
from FastML import LDL
def colize(x):
    return [x[:, i] for i in range(x.shape[1])]

class LMNN_duration(object):

    def __init__(self, 
        dim, 
        mu=0.5, K=3,
        kernelFunction=None,            # the explicit kernel
        normalizeFunction=None,         # how data will be normalized
        verbose=True,
        ):
        self.dim = dim
        self.mu = mu
        self.K = K
        self.normalizeFunction = normalizeFunction
        self.kernelFunction    = kernelFunction   
        self.verbose = verbose
        self.__Theano_build__()

    def __Theano_build__(self):
        x = T.matrix('x')
        y = T.ivector('y')
        lr = T.scalar('lr')
        M = theano.shared(value=np.eye(self.dim, dtype='float32'), name='M', borrow=True)
        triple   = [T.ivector('triple'+i) for i in ['i', 'j', 'l']]
        neighbor = [T.ivector('neighbor'+i) for i in ['i', 'j']]

        def neigbhor_loss(i, j, x, M):
            diff = x[i]-x[j]
            return diff.dot(M).dot(diff.T)
        pull_error, _ = theano.scan(
            fn = neigbhor_loss, 
            sequences=[neighbor[0], neighbor[1]],
            outputs_info=None,
            non_sequences=[x, M])
        pull_error = pull_error.sum()*self.mu

        def triple_loss(i, j, l, x, y, M):
            diffij = x[i]-x[j]
            diffil = x[i]-x[l]
            dij = diffij.dot(M).dot(diffij.T)
            dil = diffil.dot(M).dot(diffil.T)
            hinge = T.maximum(dij - dil + 1, 0)

            ydiff = (y[i] - y[l])**2
            return ydiff * hinge

        push_error, _ = theano.scan(
            fn = triple_loss, 
            sequences=[triple[0], triple[1], triple[2]],
            outputs_info=None,
            non_sequences=[x, y, M])
        push_error = push_error.sum()*(1-self.mu)
        zerocount = T.neq(y[triple[0]], y[triple[2]]).sum()
        def PSD_Project(mat):
            mat = (mat+mat.T)/2.0
            eig, eigv = linalg.eig(mat)
            eig = T.maximum(eig, 0)
            eig = T.diag(eig)
            return eigv.dot(eig).dot(eigv.T) 

        self.Tupdates = [(M, PSD_Project(M - lr*T.grad(pull_error+push_error, M)))]
        self.Ttriple = triple
        self.Tneighbor = neighbor
        self.Tlr = lr
        self.Tpull_error = pull_error
        self.Tpush_error = push_error
        self.Tx = x
        self.Ty = y
        self.TM = M
        self.Tzerocount = zerocount


#       self.Tloss = theano.function(Ttriple+Tneighbor, 
#               error,
#               givens={Td: self.D, Ty: self.y},
#               allow_input_downcast=True)

    def preprocess(self, x):
        if self.kernelFunction:
            x = self.kernelFunction(x)
        if self.normalizeFunction:
            x = self.normalizeFunction(x)
        return x

    def fit(self, x, y, testx, testy,
        tripleCount=100,
        learning_rate=1e-7,

        max_iter=100,
        reset_iter=20,
        epochs=20,

        verbose=0,
        autosaveName=None,#'temp.MLMNN',
    ):
        self.verbose = verbose
        self.tripleCount = tripleCount

        x = x.astype('float32')
        y = y.astype('int32')
        testx = testx.astype('float32')
        testy = testy.astype('int32')

        trainx = x
        x = self.preprocess(x)
        n = x.shape[0]
        if self.verbose: print 'Before pca: x.shape:', x.shape
        pca = PCA(n_components=self.dim, whiten=False)
        x = pca.fit_transform(x)
        if self.verbose: print '\tExplained variance ratio', sum(pca.explained_variance_ratio_)
        self.pca = pca

        step = theano.function(self.Ttriple+self.Tneighbor+[self.Tlr], 
                [self.Tpull_error, self.Tpush_error, self.Tzerocount],
                givens={self.Tx: x, 
                        self.Ty: y},
                updates=self.Tupdates,
                allow_input_downcast=True)
#                on_unused_input='warn')

        lr = learning_rate 
        train_acc, test_acc = self.fittest(trainx, testx, y, testy)
        for epoch_iter in xrange(epochs):
            x = self.transform(trainx)
            neighbors = self.get_neighbors(x, y)
            t = 0
            while t < max_iter:
                if t % reset_iter == 0:
                    active_set = self.get_active_set(x, y, neighbors)
                    last_error = np.inf 

                lastM = self.M
                pullError, pushError, violated = \
                    step(*(colize(active_set)+colize(neighbors)+[lr]))
                #step2(*(colize(active_set)+colize(neighbors)+[lr]))
                if verbose:
                    print 'Iter:', t 
                    print '\tViolated triples: {}/{}'.format(violated, self.tripleCount)
                    print '\tlr:', lr 
                    print '\tShare Pull: ', pullError
                    print '\tShare Push: ', pushError
                error = pullError+pushError
                
                if last_error > error:
                    lr = lr*1.02
                    last_error = error
                else:
                    lr = lr * 0.85
                    self.M = lastM

                sys.stdout.flush()

                t += 1
            train_acc, test_acc = self.fittest(trainx, testx, y, testy)

        if autosaveName:
            if verbose: print 'Auto saving ...'
            self.save(autosaveName)
        return self

    def fittest(self, trainx, testx, trainy, testy):
        trainx = self.transform(trainx)
        testx = self.transform(testx)
        train_acc, train_cfm = knn(trainx, trainx, trainy, trainy, None, self.K, cfmatrix=True)

        test_acc, test_cfm = knn(trainx, testx, trainy, testy, None, self.K, cfmatrix=True)
        if self.verbose: print 'shape: {}'.format(trainx.shape)
        if self.verbose: print 'train-acc: %.3f%%  %s'%(train_acc, ' '*30)
        if self.verbose: print 'train confusion matrix:\n {}'.format(train_cfm)
        if self.verbose:
            print 'test-acc: %.3f%%'%(test_acc)
            print 'test confusion matrix:\n {}'.format(test_cfm)
        return train_acc, test_acc

    def get_active_set(self, x, y, neighbors):
        result = []
        ijcandidates = neighbors[np.random.choice(range(neighbors.shape[0]), size=(self.tripleCount))]
        lcandidates = np.random.randint(0, x.shape[0], size=(self.tripleCount, 1) )
        ijl = np.hstack([ijcandidates, lcandidates]).astype('int32')
        return ijl 

    def get_neighbors(self, x, y):
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
                cost = (v**2).sum(1) #np.diag(v.dot(self._M).dot(v.T))
                neighbors[i] = ind[cost.argsort()[1:self.K+1]]
                COST += sum(sorted(cost)[1:self.K+1])
        if self.verbose: print 'neighbor cost: {}'.format(COST)
        #self.neighbors = neighbors

        # repeat an arange array, stack, and transpose
        #self.neighborpairs = np.array(neighborpairs, dtype='int32')
        res = np.vstack([np.repeat(np.arange(x.shape[0]), self.K), neighbors.flatten()]).T.astype('int32')
        return res 

    def transform(self, x):
        x = self.preprocess(x)

        #pca
        x = self.pca.transform(x)
        L = LDL(self.TM.get_value(), combined=True)
        return x.dot(L)

    def save(self, filename):
        cPickle.dump((self.__class__, self.__dict__), open(filename, 'w'))

    @staticmethod
    def load(filename):
        cls, attributes = cPickle.load(open(filename, 'r'))
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

    @property
    def M(self):
        return self.TM.get_value()

    @M.setter
    def M(self, value):
        self.TM.set_value(value)

    



def getD(x):
    n, m = x.shape
    d = np.zeros((m, n, n))
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            d[:, i, j] = (xi - xj)**2
    return d


# if __name__ == '__main__':
#   np.set_printoptions(linewidth=100)
#   print "Test 1, simple test"
#   X = np.array([[10, 0, 0], [20, 1, 1], [20, 2, 0], [10, 3, 1]], dtype='float32')
#   D = getD(X)
#   print D
#   y = np.array([0, 0, 1, 1], dtype='int32')
#   lmnn = LMNN_alpha()
#   lmnn.fit(D, y, max_iter=1000, lr=5e-4)

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf, precision=3)

    K = 300
    Time = 1.0
    G = 10
    print dataset_name 

    data = np.load(open(histogramFilename(K, G), 'r'))
    x, y = data['x'], data['y']
    x = x.reshape(x.shape[0], 10, -1)[:, :int(Time*10), :].reshape(x.shape[0], -1)

    newx = []
    newy = []
    for ele in x:
        acc = np.zeros((2 * K))
        ele = ele.reshape(G, -1)
        for ind, dur in enumerate(ele):
            acc = acc + dur
            newx.append(acc)
            newy.append(ind)
    x = np.array(newx)
    y = np.array(newy)

    trainx, testx, trainy, testy = \
        train_test_split(x, y, test_size=0.32, random_state=138)
    print trainx.shape
    print trainy.shape
    
#   def kernel(x, y):
#       K = np.zeros((x.shape[0], y.shape[0]))
#       for i in xrange(x.shape[0]):
#           for j in xrange(y.shape[0]):
#               K[i, j] = float((x[i]*y[i]).sum())/((x[i]+y[i])**2).sum()
#       return K
#   clf = svm.SVC(kernel='linear', C=100)
#   pred = clf.fit(trainx, trainy).predict(testx)
#   print pred
#   print testy
#   print '{}/{} = {}'.format( (pred == testy).sum(), len(testy), (pred == testy).sum()/float(len(testy)) )
#   print '{}'.format( pred - testy )

    print knn(trainx, testx, trainy, testy, None, 5, cfmatrix=True)
    from sklearn.preprocessing import normalize
    lmnn = LMNN_duration(
        dim=100, 
        mu=0.5, K=10,
        kernelFunction=None,            # the explicit kernel
        normalizeFunction=None, #normalize,         # how data will be normalized
        verbose=True)
    lmnn.fit(trainx, trainy, testx, testy,
        tripleCount=10000,
        learning_rate=1e-6,
        max_iter=5,
        reset_iter=30,
        epochs=100,
        verbose=True,
        autosaveName='duration.model',)#'temp.MLMNN',
