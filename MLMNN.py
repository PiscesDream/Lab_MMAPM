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

class MLMNN(object):
    def __init__(self, 
        granularity=10,                 # how many parts will a video clip be sliced
        K=8,                            # the neighbor count 
        mu=0.5,                         # mu in LMNN mu*PullLoss + (1-mu)*PushLoss
        lmdb=0.1,                       # L2 normalization coefficient
        pushgamma=0.1,                  # global push coefficient 
        pullgamma=0.1,                  # global push coefficient 
        shorttermMetric=True,           # short-term Metric
        longtermMetric=False,           # long-term Metric
        # preprocessing parameters (in order)
        kernelFunction=None,            # the explicit kernel
        normalizeFunction=None,         # how data will be normalized
        dim=100,                        # dim of each fragment after PCA
    ):
        self.granularity       = granularity      
        self.K                 = K                
        self.mu                = mu               
        self.lmdb              = lmdb
        self.pushgamma         = pushgamma
        self.pullgamma         = pullgamma
        self.dim               = dim              
        self.shorttermMetric   = shorttermMetric  
        self.longtermMetric    = longtermMetric   
        self.normalizeFunction = normalizeFunction
        self.kernelFunction    = kernelFunction   

        self.__theano_build__()

    def fit(self, x, y, testx=None, testy=None,
        tripleCount=100,
        learning_rate=1e-7,
        max_iter=100,
        reset_iter=20,
        epochs=20,

        verbose=0,
        autosaveName='temp.MLMNN',
    ):
        self.verbose = verbose
        self.tripleCount = tripleCount
        x = x.astype('float32')
        y = y.astype('int32')
        if testx != None:
            testx = testx.astype('float32')
            testy = testy.astype('int32')

        trainx = x
        x = self.preprocess(x)
        n = x.shape[0]
        x = x.reshape(n, self.granularity, -1)
        if self.verbose: print 'Before pca: x.shape:', x.shape
        newx = np.zeros((n, self.granularity, self.dim), dtype='float32')
        PCAs = []
        for i in xrange(self.granularity):
            pca = PCA(n_components=self.dim, whiten=False)
            newx[:, i, :] = pca.fit_transform(x[:, i, :])
            if self.verbose: print '\tPCA[%d]: Explained variance ratio' % i, sum(pca.explained_variance_ratio_)
            PCAs.append(pca)
        x = newx
        self.PCAs = PCAs


        givens = {self.Ty: y}
        if self.shorttermMetric:
            givens.update([(self.Tx[i], x[:, i, :]) for i in range(self.granularity)])
        if self.longtermMetric:
            stackx = x.copy()
            for i in xrange(1, self.granularity):
                stackx[:, i, :] += stackx[:, i-1, :]
            givens.update([(self.Tstackx[i], stackx[:, i, :]) for i in range(self.granularity)])

        step = theano.function(
            self.Ttriple+self.Tneighbor+[self.Tlr],
            [T.stack(self.TpullErrors), \
             T.stack(self.TpushErrors), \
             T.stack(self.TregError), \
             self.nonzerocount, \
             T.stack(self.globalPullError), \
             T.stack(self.globalPushError), \
            ],
            updates = self.Tupdates,
            givens = givens)

#       step2 = theano.function(
#           self.Ttriple+self.Tneighbor+[self.Tlr],
#           [],
#           updates = self.Tupdates2,
#           givens = givens)

#       debug = theano.function(
#           self.Ttriple+self.Tneighbor,
#           [self.debug1, self.debug2],
#           givens = givens,
#           on_unused_input='warn')

        mcount = (self.shorttermMetric+self.longtermMetric)*self.granularity;
        lr = np.array([learning_rate]*mcount, dtype='float32')
        train_acc, test_acc = self.fittest(trainx, testx, y, testy)
        for epoch_iter in xrange(epochs):
            x = self.transform(trainx)
            neighbors = self.get_neighbors(x, y)
            t = 0
            while t < max_iter:
                if t % reset_iter == 0:
                    active_set = self.get_active_set(x, y, neighbors)
                    last_error = np.array([np.inf]*(mcount))

                if verbose: print 'Iter: {} lr: {} '.format(t, lr)
                pullError, pushError, regError, \
                    violated, globalPullError, globalPushError = \
                        step(*(colize(active_set)+colize(neighbors)+[lr]))
                #step2(*(colize(active_set)+colize(neighbors)+[lr]))
                print 'Violated triples: {}/{}'.format(violated, self.tripleCount)
                if verbose:
                    print pullError
                    print pushError
                    print regError
                    print globalPullError
                    print globalPushError
                error = (pullError+pushError+regError+globalPullError+globalPushError).flatten()
                
#               d1, d2 = debug(*(colize(active_set)+colize(neighbors)))
#               import pdb
#               pdb.set_trace()

                lr = lr*1.05*(last_error>error) + lr*0.75*(last_error<=error) 
                last_error = error
                t += 1
            train_acc, test_acc = self.fittest(trainx, testx, y, testy)

        if autosaveName:
            if verbose: print 'Auto saving ...'
            self.save(autosaveName)
        return self

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

    def save(self, filename):
        cPickle.dump((self.__class__, self.__dict__), open(filename, 'w'))

    @staticmethod
    def load(filename):
        cls, attributes = cPickle.load(open(filename, 'r'))
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

    def preprocess(self, x):
        if self.kernelFunction:
            x = self.kernelFunction(x)
        if self.normalizeFunction:
            x = self.normalizeFunction(x)
        return x

    def transform(self, x, M=-1, nostack=False, alpha=None):
        print x.shape
        if M == -1: M = self.granularity
        if alpha == None: 
            alpha = self.Talpha[M-1].get_value().flatten()# np.ones((M,))
            print alpha
        x = self.preprocess(x)

        #pca
        n = x.shape[0]
        x = x.reshape(n, M, -1)
        newx = np.zeros((n, M, self.dim))
        for i in xrange(M):
            newx[:, i, :] = self.PCAs[i].transform(x[:, i, :])
        x = newx

        stackx = np.zeros((x.shape[0], self.dim))
        newx = []
        for i in xrange(M):
            if self.shorttermMetric:
                L = LDL(self.Tstm[i].get_value(), combined=True)
                newx.append(x[:, i, :].dot(L)*np.sqrt(alpha[i]))
            #print 'MS[{}] Size: {} rank: {} segment: {}'.\
                #format(i, self.lM[i].get_value().shape, \
                #nplinalg.matrix_rank(self.lM[i].get_value()), newx[-1].shape)

            if self.longtermMetric:
                stackx += x[:, i, :]
        if self.longtermMetric: newx.append(stackx.dot(LDL(self.Tltm[-1].get_value(), combined=True)) )
        if nostack:
            newx = np.array(newx)
        else:
            newx = np.hstack(newx)
            newx[np.isnan(newx)] = 0
            newx[np.isinf(newx)] = 0
        return newx

    def fittest(self, trainx, testx, trainy, testy, G=-1, alpha=None):
        trainx = self.transform(trainx, G, alpha=alpha)
        if testx != None: testx = self.transform(testx, G, alpha=alpha)
        train_acc, train_cfm = knn(trainx, trainx, trainy, trainy, None, self.K, cfmatrix=True)

        if testx != None: test_acc, test_cfm = knn(trainx, testx, trainy, testy, None, self.K, cfmatrix=True)
        if self.verbose: print 'shape: {}'.format(trainx.shape)
        if self.verbose: print 'train-acc: %.3f%%  %s'%(train_acc, ' '*30)
        if self.verbose: print 'train confusion matrix:\n {}'.format(train_cfm)
        if testx != None and self.verbose:
            print 'test-acc: %.3f%%'%(test_acc)
            print 'test confusion matrix:\n {}'.format(test_cfm)
        if testx != None: 
            return train_acc, test_acc
        else:
            return test_acc
        
    def __theano_build__(self):
        self.Tx        = [T.matrix('x{}'.format(i), dtype='float32') for i in range(self.granularity)]
        self.Tstackx   = [T.matrix('stackx{}'.format(i), dtype='float32') for i in range(self.granularity)]
        self.Ty        = T.ivector('y') 
        self.Tlr       = T.vector('lr', dtype='float32')
        self.Ttriple   = [T.ivector('triple'+x) for x in ['i', 'j', 'l']]
        self.Tneighbor = [T.ivector('neighbor'+x) for x in ['i', 'j']]
        self.Talpha    = [theano.shared(value=np.ones((i)).astype('float32')/i, name='alpha[{}]'.format(i), borrow=True) \
                                for i in range(1, self.granularity+1) ]
#       self.Talpha[1].set_value(np.array([[[0.3],[0.7]]], dtype='float32'))

        def PSD_Project(mat):
            mat = (mat+mat.T)/2.0
            eig, eigv = linalg.eig(mat)
            eig = T.maximum(eig, 0)
            eig = T.diag(eig)
            return eigv.dot(eig).dot(eigv.T) 

        # list of metrics
        stm = [] # short-term metrics
        ltm = [] # long-term metrics
        # summing up 
        updates = []
        pullErrors = []
        pushErrors = []
        Dneighbor = []
        Dtripleij = []
        Dtripleil = []
        Error = 0.0
        # build calc graph
        #index = 0
        if self.shorttermMetric:
            for i in xrange(self.granularity):
                M = theano.shared(value=np.eye(self.dim, dtype='float32'), name='short-termM[{}]'.format(i), borrow=True)
                pullError, pushError, dneighbor, dtripleij, dtripleil = \
                    self.__theano_shorttermError(M, i, self.Tx[i])
                pullError *= (1-self.mu); pushError *= self.mu
                #update = (M, PSD_Project(M - self.Tlr[index] * T.grad(pullError + pushError, M)) )
                #index += 1

                stm.append(M)
                pushErrors.append(pushError)
                pullErrors.append(pullError)
                Dneighbor.append(dneighbor)  
                Dtripleij.append(dtripleij) 
                Dtripleil.append(dtripleil)
                #updates.append(update)
        if self.longtermMetric:
            for i in xrange(self.granularity):
                pass
#               M = theano.shared(value=np.eye(self.dim, dtype='float32'), name='long-termM[{}]'.format(i), borrow=True)
#               if i == 0:
#                   pullerror, pusherror = self.__theano_longtermError(M, i, None, self.Tstackx[i])
#               else:
#                   pullerror, pusherror = self.__theano_longtermError(M, i, ltm[-1], self.Tstackx[i])
#               pullError *= (1-self.mu); pushError *= self.mu
#               #update = (M, PSD_Project(M - self.Tlr[index] * T.grad(pullError + pushError, M)) )
#               #index += 1

#               ltm.append(M)
#               pushErrors.append(pushError)
#               pullErrors.append(pullError)
#               #updates.append(update)

        regError = [] 
        for ele in stm+ltm:
            regError.append(self.lmdb * (ele**2).sum())

        globalPullError = [] 
        globalPushError = [] 

        for i in range(1, self.granularity+1):
            globalPullError.append(self.pullgamma*\
                                (self.Talpha[i-1].dimshuffle(0, 'x') * T.stacklists(Dneighbor[:i])).sum() )
            globalPushError.append(self.pushgamma*\
                                T.maximum((self.Talpha[i-1].dimshuffle(0, 'x') * T.stacklists(Dtripleij[:i])).sum(1)-\
                                          (self.Talpha[i-1].dimshuffle(0, 'x') * T.stacklists(Dtripleil[:i])).sum(1)+1, 0).sum() )

        Error = T.sum(pushErrors) + T.sum(pullErrors) + T.sum(regError) + \
            T.sum(globalPullError) + T.sum(globalPushError)

#       i = 2
#       self.debug1 = self.Talpha[i-1]
#       self.debug2 = T.grad(Error, self.Talpha[i-1]) 
#       self.debug1 = T.maximum((self.Talpha[i-1] * T.stack(Dtripleij[:i])).sum(1)-(self.Talpha[i-1] * T.stack(Dtripleil[:i])).sum(1)+1, 0)
#       self.debug2 = (self.Talpha[i-1] * T.stack(Dneighbor[:i]))

        self.Tupdates = \
            [(ele, PSD_Project(ele - self.Tlr[index] * T.grad(Error, ele)))
                for index, ele in enumerate(stm+ltm)] 
        for ind, ele in enumerate(self.Talpha):
            temp = ele - self.Tlr[ind] * T.grad(Error, ele)
            self.Tupdates.append( (ele, temp/temp.sum()) )

        self.Tstm = stm
        self.Tltm = ltm
        self.TpullErrors = pullErrors
        self.TpushErrors = pushErrors
        self.TregError = regError 
        self.globalPullError = globalPullError 
        self.globalPushError = globalPushError 

    def __theano_shorttermError(self, targetM, i, x):
        pull_error = 0.
        ivectors = x[self.Tneighbor[0]]
        jvectors = x[self.Tneighbor[1]]
        diffv = ivectors - jvectors
        Dneighbor = linalg.diag(diffv.dot(targetM).dot(diffv.T))
        pull_error = T.sum(Dneighbor)
        
        push_error = 0.0
        ivectors = x[self.Ttriple[0]]
        jvectors = x[self.Ttriple[1]]
        lvectors = x[self.Ttriple[2]]
        diffij = ivectors - jvectors
        diffil = ivectors - lvectors
        lossij = diffij.dot(targetM).dot(diffij.T)
        lossil = diffil.dot(targetM).dot(diffil.T)
        mask = T.neq(self.Ty[self.Ttriple[0]], self.Ty[self.Ttriple[2]])
        Dtripleij = linalg.diag(lossij)
        Dtripleil = linalg.diag(lossil)
        push_error = (mask*T.maximum(Dtripleij  - Dtripleil + 1, 0)).sum()

        self.nonzerocount = (mask*linalg.diag(T.gt(lossij - lossil + 1, 0))).sum()

#       print np.sqrt((i+1.0)/self.M)
#       pull_error = pull_error * np.sqrt((i+1.0)/self.M)
#       push_error = push_error * np.sqrt((i+1.0)/self.M)

        return pull_error, push_error, Dneighbor, Dtripleij, Dtripleil

    def __theano_longtermError(self, targetM, i, lastM):
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


    def get_DMatrix(self, x, y=None, M=-1):
        x = self.transform(x, M=M, nostack=True)
        if y!=None:
            y = self.transform(y, M=M, nostack=True)
        else:
            y = x
        d = np.zeros((self.granularity, x.shape[1], y.shape[1]))
        # x = (10, n, xx)
        # y = (10, m, xx)
        for g in range(self.granularity):
            for i in range(x.shape[1]):
                for j in range(y.shape[1]):
                    d[g, i, j] = ((x[g, i] - y[g, j])**2).sum()
        d[np.isnan(d)] = 0
        return d


def train():
    mlmnn = MLMNN(granularity=int(Time*groups), 
            K=5, 
            mu=0.5, lmdb=0.05, pushgamma=2.0, pullgamma=2.0, 
            dim=150, 
            normalizeFunction=normalize) 
#   try:
    if True:
        mlmnn.fit(trainx, trainy, testx, testy, 
            tripleCount=10000, learning_rate=0.5, 
            max_iter=10, reset_iter=5, epochs=20, 
            verbose=True)
#   except:
#       mlmnn.save('temp.MLMNN')
#       print 'early break'

#   [ 0.1     0.2     0.3     0.4     0.5     0.6     0.7     0.8     0.9     1.0  ]
#   [ 50.746  73.134  86.94   94.776  98.134  99.627  100.    100.    99.627  99.627]
#   [ 18.939  36.364  53.788  60.606  74.242  78.788  74.242  78.03   79.545  80.303]

#   [ 47.388  76.493  88.433  94.776  99.254  99.254  99.627  99.627  99.627  99.627]
#   [ 18.939  34.091  46.97   59.091  67.424  75.758  78.03   80.303  84.091  81.818]

#   [ 51.119  73.134  90.299  93.657  98.881  98.881  99.254  99.627  99.627  99.627]
#   [ 25.     39.394  50.758  65.909  74.242  76.515  77.273  84.091  81.818  81.818]

#   [ 52.812  67.812  79.375  91.875  98.125  99.062  99.688  99.375  99.062  99.062]
#   [ 22.5    27.5    42.5    56.25   73.75   81.25   83.75   85.     83.75   85.  ]

#   [ 50.     66.944  79.444  87.222  95.278  98.889  98.889  99.167  98.611  98.889]
#   [ 27.5  42.5  62.5  62.5  72.5  82.5  80.   85.   85.   85. ]

# rs = 133
#   [ 45.896  59.701  73.507  83.582  88.806  93.284  94.03   92.537  91.418  91.418]
#   [ 23.485  35.606  53.788  61.364  67.424  73.485  75.758  76.515  77.273  77.273]

# rs = 134
#   [ 48.507  69.776  82.836  91.045  97.015  98.881  98.134  97.388  97.015  96.642]
#   [ 21.212  28.788  45.455  57.576  65.909  75.758  73.485  74.242  71.97   71.212]

#   [ 53.358  75.373  88.06   96.269  99.254  100.    100.    100     100.    99.627]
#   [ 21.97   35.606  53.03   60.606  68.939  72.727  73.485  75.     71.97   71.97 ]

#     multi-task
#   [ 46.269  72.761  86.94   92.537  98.507  99.627  99.254  99.627  99.254  98.881]
#   [ 22.727  29.545  49.242  62.121  67.424  76.515  75.758  73.485  73.485  75.   ]

# rs = 138

#   [ 43.75   56.618  73.162  80.515  86.765  91.544  92.647  92.647  90.809  90.809]
#   [ 17.969  39.062  51.562  55.469  70.312  77.344  82.031  79.688  77.344  78.906]

#       K=5, mu=0.5, lmdb=0.05, dim=100, 
#       tripleCount=10000, learning_rate=1.0, max_iter=10, reset_iter=5, epochs=5, 
#   [ 50.278  62.222  74.722  84.167  91.667  95.556  94.722  95.833  94.444  94.167]
#   [ 22.5    32.5    45.     60.     72.5    85.     82.5    82.5    82.5    82.5]

#       with global push and pull
#   [ 47.388  64.552  80.97   88.433  94.03   96.269  97.761  97.388  97.015  97.015]
#   [ 24.242  38.636  49.242  60.606  67.424  71.97   75.758  78.03   81.818  82.576]

#       multi-task
#   [ 56.716  75.746  90.672  93.657  97.388  98.881  99.627  99.627  99.627  99.254]
#   [ 25.758  40.152  49.242  59.848  69.697  78.03   77.273  78.788  79.545  80.303]

#       alpha first time
#   [ 40.299  46.642  61.567  63.806  69.776  60.821  66.791  68.657  71.269  72.015]
#   [ 18.182  31.061  32.576  34.091  39.394  36.364  43.939  43.182  43.939  43.939]

#     32.81   36.72   53.90   59.38   67.97   63.28   68.75   75.00   75.78   79.69


def loadtest(trainx, testx, trainy, testy, As=None):
    mlmnn = MLMNN.load('temp.MLMNN')

    acc = []
    for Time2 in np.linspace(0.1, 1, 10):
        f = lambda x: x.reshape(x.shape[0], 10, -1)[:, :int(Time2*10), :].reshape(x.shape[0], -1)
        ttrainx = f(trainx)
        ttestx = f(testx)
        if As:
            acc.append(mlmnn.fittest(ttrainx, ttestx, trainy, testy, int(Time2*10), alpha=As[int(Time2*10)-1]))
        else:
            acc.append(mlmnn.fittest(ttrainx, ttestx, trainy, testy, int(Time2*10)))

    print acc
    acc = np.array(acc)
    print acc[:,0]
    print acc[:,1]   
    return acc

def testmcml():
    mlmnn = MLMNN.load('temp.MLMNN') # randomstate=32
    trainx = mlmnn.transform(trainx)
    testx = mlmnn.transform(testx)

    from FastML import MCML, KNN
    knn = KNN(n_neighbors=5)
    mcml = MCML()
    pred = knn.fit(trainx, trainy).predict(testx)
    print '{}/{}'.format( (pred == testy).sum(), len(testy) )
    for _ in range(100):
        trainx = mcml.fit(trainx, trainy, max_iter=1, lr=1e-7).transform(trainx)
        testx = mcml.transform(testx)

        pred = knn.fit(trainx, trainy).predict(testx)
        print '{}/{}'.format( (pred == testy).sum(), len(testy) )

if __name__ == '__main__':
    from sklearn.cross_validation import train_test_split
    from sklearn.preprocessing import normalize
    import cPickle
    np.set_printoptions(linewidth=np.inf, precision=3)
    theano.config.exception_verbosity = 'high'

    K = 200
    Time = 1.0
    groups = 10
    from names import featureDir, globalcodedfilename
    if groups == 10:
        x, y = cPickle.load(open('{}/[K={}][T={}]BoWInGroup.pkl'.format(featureDir, K, 1.0), 'r'))
    else:
        x, y = cPickle.load(open(globalcodedfilename(K, 1.0, groups), 'r'))
    x = x.reshape(x.shape[0], 10, -1)[:, :int(Time*10), :].reshape(x.shape[0], -1)
    print y
    trainx, testx, trainy, testy = \
        train_test_split(x, y, test_size=0.33, random_state=138)
    print trainx.shape
    print trainy.shape

    train()
    acc1 = loadtest(trainx, testx, trainy, testy)

#   [ 56.343   73.134   88.06    94.776   98.881  100.      99.627  100.     100.      99.627]
#   [ 17.424  33.333  43.182  52.273  65.909  75.     76.515  78.03   75.     76.515]

#   testmcml()


