import numpy as np
from sklearn.decomposition import PCA
from common import load_mnist, knn

import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg
import numpy.linalg as nplinalg
from Metric.LDL import LDL
import sys

class LMNN(object):
    def __init__(self, K=3, mu=0.5, maxS=100, dim=None):
        import sys
        sys.setrecursionlimit(10000)

        self.K = K
        self.mu = mu
        self.maxS = maxS 

        self._x = T.matrix('_x', dtype='float32')
        self._y = T.ivector('_y') 
        self._lr = T.scalar('_lr')
        self._set = T.imatrix('_set')
        self._neighborpairs = T.imatrix('_neighborpairs')
        self.dim = dim # dim is used when M is not trained
        pass

    def build(self, dim=None):
        if dim: self.dim = dim
        if not self.dim:
            raise Exception("Dim is not defined")

        self.M = M = theano.shared(
                value=np.eye(self.dim, dtype='float32'),
                name='M',
                borrow=True)
        self._init_neighbors()
        self._init_error()
        self.updates = [(M, self.M - self._lr * T.grad(self.error, M) )]


    def fit(self, x, y, verbose=False, lr=1e-7, max_iter=100, reset=20, tol=1e-7):
        self.n, self.m = x.shape
        self.x, self.y= x, y 
        x = np.asarray(x, dtype='float32')
        y = np.asarray(y, dtype='int')

        try:
            M = self.M
        except:
            if verbose: print 'Building model ...'
            self.trained = True

            self.build(self.m)
            M = self.M
        self.train_model = theano.function([self._set, self._neighborpairs, self._lr], [self.pull_error, self.push_error],
            updates = self.updates,
            givens = {
                self._x: x 
                })

        if verbose: print 'training ...'
        t = 0
        last_error = np.inf
        while t < max_iter:
            if t % reset == 0:
                active_set = self._get_active_set()

            # train
            pull_error, push_error = self.train_model(active_set, self.neighborpairs, lr)
            pull_error = pull_error.sum() * (1 - self.mu)
            push_error = push_error.sum() * (self.mu)
            if verbose: 
                print 'Iter={} error={} update={} lr={}'.\
                format(t, (pull_error, push_error), pull_error+push_error-last_error, lr)
            else:
                print 'Iter={} error={} update={} lr={}\r'.\
                format(t, (pull_error, push_error), pull_error+push_error-last_error, lr),
                sys.stdout.flush()

            # projection (calculation loss lead to a singular(non-symmetric) matrix
            # numpy can handle complex better than theano
            _M = M.get_value() 
            M.set_value(np.array(self._numpy_project_sd(_M), dtype='float32'))
            #print 'rank={}'.format(nplinalg.matrix_rank(_M))

            # update lr
            error = pull_error.sum() + push_error.sum()
            lr = lr * 1.01 if last_error>error or t%reset==0 else lr * 0.5
            if t%reset!=0 and abs(error-last_error)<tol: break


            last_error = error
            t += 1

        self.M = M
        return self

    @property
    def _M(self):
        try:
            return self.get_M()
        except:
            return np.eye(self.dim) 

    def get_M(self):
        return self.M.get_value()

    @property
    def L(self):
        return LDL(self._M, combined=True)

    def _get_active_set(self):
        result = []
        candidates = np.random.randint(0, self.n, size=(self.maxS, 2))
        jcandidates = np.random.randint(0, self.K, size=(self.maxS) )
        for (i, l), j in zip(candidates, jcandidates):
            if self.y[i] == self.y[l]: continue
            result.append((i, j, l)) #vij, vil ))
        return np.array(result, dtype='int32')

    def _CIJ(self, i, j):
        diff = self._x[i]-self._x[j]
        return T.dot(T.dot(diff, self.M), diff).sum()
        #return linalg.trace(self.M.dot(T.outer(diff, diff)))

    def _init_error(self):
        pull_error = 0.
        ivectors = self._x[self._neighborpairs[:,0]]
        jvectors = self._x[self._neighborpairs[:,1]]
        diffv = ivectors - jvectors
        pull_error = linalg.trace(diffv.dot(self.M).dot(diffv.T))

        push_error = 0.0
        ivectors = self._x[self._set[:,0]]
        jvectors = self._x[self._set[:,1]]
        lvectors = self._x[self._set[:,2]]
        diffij = ivectors - jvectors
        diffil = ivectors - lvectors
        lossij = diffij.dot(self.M).dot(diffij.T)
        lossil = diffil.dot(self.M).dot(diffil.T)
        push_error = linalg.trace(T.maximum(lossij - lossil + 1, 0))

        self.pull_error = pull_error
        self.push_error = push_error
        self.error = (1-self.mu) * pull_error +\
                      self.mu * push_error

    def _init_neighbors(self):
        neighbors = np.zeros((self.n, self.K), dtype=int)
        yset = set(self.y)
        COST = 0.0
        for c in yset:
            mask = self.y==c
            ind = np.arange(self.n)[mask]
            x = self.x[mask]

            for i in xrange(self.n):
                if self.y[i] != c: continue
                v = self.x[i] - x
                cost = np.diag(v.dot(self._M).dot(v.T))
                neighbors[i] = ind[cost.argsort()[1:self.K+1]]
                COST += sum(sorted(cost)[1:self.K+1])
        print 'neighbor cost: {}'.format(COST)
        self.neighbors = neighbors

        neighborpairs = []
        for i in xrange(self.n):
            for j in neighbors[i]:
                neighborpairs.append([i,j])
        self.neighborpairs = np.array(neighborpairs, dtype='int32')

    def _project_sd(self, mat):
        eig, eigv = linalg.eig(mat)
        eig = T.diag(T.maximum(eig, 0))
        return eigv.dot(eig).dot(linalg.matrix_inverse(eigv))

    def _numpy_project_sd(self, mat):
        eig, eigv = nplinalg.eig(mat)
        eig[eig < 0] = 0
        eig = np.diag(eig)
        return eigv.dot(eig).dot(nplinalg.inv(eigv))

if __name__ == '__main__':
    theano.config.exception_verbosity='high'
    (trainx, trainy), (testx, testy) = load_mnist(percentage=0.01, skip_valid=True)

    pca = PCA(whiten=True)
    pca.fit(trainx)
    components, variance = 0, 0.0
    for components, ele in enumerate(pca.explained_variance_ratio_):
        variance += ele
        if variance > 0.90: break
    components += 1
    print 'n_components=%d'%components
    pca.set_params(n_components=components)
    pca.fit(trainx)

    trainx = pca.transform(trainx)
    testx = pca.transform(testx)

    K = 3
    dim = components
    lmnn = LMNN(K=K, mu=0.5, dim=dim, maxS=100)
    #trainx, testx, trainy, testy = train_test_split(X, Y, test_size=0.66)
    knn(trainx, testx, trainy, testy, np.eye(dim), K)
    knn(trainx, testx, trainy, testy, np.random.rand(dim, dim), K)
    for _x in xrange(50):
        lmnn.fit(trainx, trainy, lr=2e-5, max_iter=10, reset=20, verbose=True)
        lmnn._init_neighbors()

        train_acc = knn(trainx, trainx, trainy, trainy, lmnn.M.get_value(), K)
        test_acc = knn(trainx, testx, trainy, testy, lmnn.M.get_value(), K)
        print 'train_acc={} test_acc={}'.format(train_acc, test_acc)

