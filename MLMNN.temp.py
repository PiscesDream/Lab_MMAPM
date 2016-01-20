# Multi-stage LMNN
# test UTI1 with Leave-One-Out

import numpy as np
import cPickle
from sklearn.cross_validation import LeaveOneOut
from names import featureDir, globalcodedfilename

from MLMNN import MLMNN
import sys

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf, precision=3)

    K = 400
    Time = 0.1
    x, y= cPickle.load(open(globalcodedfilename(K, Time), 'r'))
#    trainx, testx, trainy, testy = train_test_split(x, y, test_size=1./60.) 
#    trainx, testx, trainy, testy = cPickle.load(open('{}/[K={}][T={}]BoWInGroup.pkl'.format(featureDir, K, Time),'r'))
#    print trainx.shape
#    print trainy.shape
   
    train_acc, test_acc = [], []
    loo = LeaveOneOut(x.shape[0]) 
    for train_index, test_index in loo:
        trainx, testx = x[train_index], x[test_index]
        trainy, testy = y[train_index], y[test_index]

        mlmnn = MLMNN(M=1, K=5, mu=0.5, dim=150, lmbd=0.5, 
                        normalize_axis=1, 
                        kernelf=None, localM=True, globalM=False) 
        _, train_res, test_res = mlmnn.fit(trainx, trainy, testx, testy, \
            maxS=10000, lr=5, max_iter=80, reset=10, rounds=1,
            Part=None, 
            verbose=False)

        print train_res, test_res
        sys.stdout.flush()

        train_acc.append(train_res)
        test_acc.append(test_res)
    print np.mean(train_acc)
    print np.mean(test_acc)

