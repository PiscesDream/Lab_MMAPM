# Multi-stage LMNN
# test UTI1 with Leave-One-Out

import numpy as np
import cPickle
from sklearn.cross_validation import LeaveOneOut
from names import *
from sklearn.preprocessing import normalize

from MLMNN import MLMNN
import sys

def loadtest(trainx, testx, trainy, testy, model=None):
    if model:
        mlmnn = model
    else:
        print 'load temp.MLMNN'
        mlmnn = MLMNN.load('temp.MLMNN')

    acc = []
    for Time2 in np.linspace(0.1, 1, 10):
        f = lambda x: x.reshape(x.shape[0], 10, -1)[:, :int(Time2*10), :].reshape(x.shape[0], -1)
        ttrainx = f(trainx)
        ttestx = f(testx)
        acc.append(mlmnn.fittest(ttrainx, ttestx, trainy, testy, int(Time2*10)))

    print acc
    acc = np.array(acc)
    print acc[:,0]
    print acc[:,1]   
    return acc



if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf, precision=3)

    K = 400
    Time = 1.0
    G = 10
    data = np.load(open(histogramFilename(K, G), 'r'))
    x, y = data['x'], data['y']    
    x = x.reshape(x.shape[0], -1)
    print x.shape
#    trainx, testx, trainy, testy = train_test_split(x, y, test_size=1./60.) 
#    trainx, testx, trainy, testy = cPickle.load(open('{}/[K={}][T={}]BoWInGroup.pkl'.format(featureDir, K, Time),'r'))
#    print trainx.shape
#    print trainy.shape
   
    train_acc, test_acc = [], []
    loo = LeaveOneOut(x.shape[0]) 
    for train_index, test_index in loo:
        trainx, testx = x[train_index], x[test_index]
        trainy, testy = y[train_index], y[test_index]

        mlmnn = MLMNN(granularity=int(Time*G),
                K=len(set(y)),
                mu=0.5, lmdb=0.25, gamma=5.00,
                dim=110, alpha_based=0.0,
                normalizeFunction=normalize)
        mlmnn.fit(trainx, trainy, testx, testy,
            tripleCount=10000, 
            learning_rate=0.5, alpha_learning_rate=1e-5,
            max_iter=2, reset_iter=5, epochs=15,
            verbose=True) 
        
        acc = loadtest(trainx, testx, trainy, testy, mlmnn)
        sys.stdout.flush()

        train_acc.append(acc[:, 0])
        test_acc.append(acc[:, 1])
    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)
    print np.mean(train_acc)
    print np.mean(test_acc)

    cPickle.dump((train_acc, test_acc), open('{}/log11.pkl'.format('UTI'), 'w'))

