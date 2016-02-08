from MLMNN import MLMNN
import theano
import numpy as np
from Metric.common import knn

class MultiMLMNN(object):

    def __init__(self, n, spliter, **kwargs):
        self.n = n
        self.spliter = spliter
        self.K = kwargs['K'] 
        self.MLMNNs = [MLMNN(**kwargs) for i in range(n)]

    def fit(self, trainx, trainy, testx, testy, **kwargs):
        for i in range(self.n):
            print 'training {} ...'.format(i)
            self.MLMNNs[i].fit(self.spliter(trainx, i), trainy, self.spliter(testx, i), testy, **kwargs)
        print 'done'

    def transform(self, data, ranges=None, **kwargs):
        if ranges==None: ranges = range(self.n)
        x = [self.MLMNNs[i].transform(self.spliter(data, i), **kwargs) for i in [0, 1]]
        return np.hstack(x)
     
    def fittest(self, trainx, testx, trainy, testy, test_range=None, **kwargs):
        if test_range==None: test_range = [None]
        for ranges in test_range:
            trainx = self.transform(trainx, ranges, **kwargs)
            testx = self.transform(testx, ranges, **kwargs)
            train_acc, train_cfm = knn(trainx, trainx, trainy, trainy, None, self.K, cfmatrix=True)

            if testx != None: test_acc, test_cfm = knn(trainx, testx, trainy, testy, None, self.K, cfmatrix=True)
            print 'shape: {}'.format(trainx.shape)
            print 'train-acc: %.3f%%  %s'%(train_acc, ' '*30)
            print 'train confusion matrix:\n {}'.format(train_cfm)
            print 'test-acc: %.3f%%'%(test_acc)
            print 'test confusion matrix:\n {}'.format(test_cfm)
            return train_acc, test_acc

    def save(self, filename):
        cPickle.dump((self.__class__, self.__dict__), open(filename, 'w'))

    @staticmethod
    def load(filename):
        cls, attributes = cPickle.load(open(filename, 'r'))
        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

 

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
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.33, random_state=131)
    print trainx.shape
    print trainy.shape


    f = lambda x, f, t: x.reshape(x.shape[0], groups, -1)[:, :, f:t].reshape(x.shape[0], -1)
    def spliter(x, index):
        if index == 0:
            return x.reshape(x.shape[0], x.shape[1]/K/2, -1)[:, :, :K].reshape(x.shape[0], -1)
        else:
            return x.reshape(x.shape[0], x.shape[1]/K/2, -1)[:, :, K:].reshape(x.shape[0], -1)


#   mmlmnn = MultiMLMNN(2, spliter=spliter,
#           granularity=int(Time*groups), K=5, mu=0.5, dim=100, 
#           normalizeFunction=normalize)
#   mmlmnn.fit(trainx, trainy, testx, testy,
#              tripleCount=10000, learning_rate=2.0, max_iter=10, 
#              reset_iter=5, epochs=5, verbose=True, autosaveName=None)
#   mmlmnn.save('multiMLMNN.model')
    mmlmnn = MultiMLMNN.load('multiMLMNN.model')
    
    acc = []
    for Time2 in np.linspace(0.1, 1, 10):
        f = lambda x: x.reshape(x.shape[0], 10, -1)[:, :int(Time2*10), :].reshape(x.shape[0], -1)
        ttrainx = f(trainx)
        ttestx = f(testx)
        acc.append(mmlmnn.fittest(ttrainx, ttestx, trainy, testy, M=int(Time2*10)))

    print acc
    acc = np.array(acc)
    print acc[:,0]
    print acc[:,1]   
