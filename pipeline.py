from duration import LMNN_duration 
from getBoW import _getHistogram
import sys
sys.path.append('/home/shaofan/Projects') 
from FastML import KNN

from MLMNN import MLMNN
from sklearn.cross_validation import train_test_split
import numpy as np
from names import *

def getHistogram(data, ind, K, G, threshold=None):
    x = []
    for i in ind:
        lx = data['lx{}'.format(i)]
        lx = _getHistogram(lx[0], lx[1], K, G, threshold)
        # dim = (G, K)

        gx = data['gx{}'.format(i)]
        gx = _getHistogram(gx[0], gx[1], K, G, threshold)
        # dim = (G, K)

        x.append( np.hstack([lx, gx]).astype('float32') )# dim=(G, 2*K)
    return np.array(x, dtype='float32')

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

def expand(x, y, K, G): 
    newx = []
    newy = []
    for ele in x:
        acc = np.zeros((2 * K))
        ele = ele.reshape(G, -1)
        for ind, dur in enumerate(ele):
            acc = acc + dur
            newx.append(acc)
            newy.append(ind+1)
    return np.array(newx), np.array(newy)

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf, precision=3)
    print dataset_name 

    K = 300
    Time = 1.0
    G = 10
    n_neighbor = 10

    data = np.load(open(histogramFilename(K, G), 'r'))
    rawdata = np.load(open(codedFilename(K), 'r'))
#   assert all([(x[i]==getHistogram(rawdata, [i], K=K, G=G)).all() for i in range(n)])
#   assert all(data['y']==rawdata['y'])

    x, y = data['x'], data['y']
    n = x.shape[0]
    x = x.reshape(n, -1)
    trainx, testx, trainy, testy = \
        train_test_split(x, y, test_size=0.32, random_state=32)
    train_ind, test_ind = \
        train_test_split(range(n), range(n), test_size=0.32, random_state=32)[:2]
#   assert ((trainx==x[train_ind]).all())
#   assert ((testx==x[test_ind]).all())

    duration_model = LMNN_duration.load('./BIT/models/[K=300][rs=32]duration.model')
    duration_trainx, duration_trainy = expand(trainx, trainy, K, G)
    duration_testx, duration_testy = expand(testx, testy, K, G)
    duration_trainx = duration_model.transform(duration_trainx)
    duration_testx = duration_model.transform(duration_testx)
    duration_knn = KNN(n_neighbor).fit(duration_trainx, duration_trainy)
    print duration_trainx.shape
    print duration_trainy.shape
    prediction = duration_knn.predict(duration_testx)
    print prediction.tolist()

    mlmnn = MLMNN.load('./BIT/models/[K=300][rs=32]mlmnn.model')
    mlmnn_knns = []
    trunc = lambda x: x.reshape(x.shape[0], G, -1)[:, :g, :].reshape(x.shape[0], -1)
    for g in range(1, G+1):
        mlmnn_knn = KNN(mlmnn.K).fit(mlmnn.transform(trunc(trainx), g), trainy)
        mlmnn_knns.append(mlmnn_knn)

    for g in range(1, G+1):
#       print '='*10
#       acc = 0.
#       for count, ind in enumerate(test_ind):
#           ttest = mlmnn.transform(trunc(x[ind:ind+1]), g)
#           if mlmnn_knns[g-1].predict(ttest)[0]==y[ind]:
#               acc += 1
#           print acc/(count+1)

        ttestx = mlmnn.transform(trunc(testx), g)
        print ttestx.shape
        acc = (mlmnn_knns[g-1].predict(ttestx)==testy).sum()
        print float(acc)/len(testy)
        print testy


    accpred = np.zeros((G))
    accpredonGreal = np.zeros((G))
    for count, ind in enumerate(test_ind):
        yi = y[ind]
        for g, threshold in enumerate(np.linspace(0, 1, G+1)[1:]):
            xi = getHistogram(rawdata, [ind], K=K, G=1, threshold=threshold).reshape(1, -1)
            Gpredict = duration_knn.predict(duration_model.transform(xi))[0]

            xi = getHistogram(rawdata, [ind], K=K, G=Gpredict, threshold=threshold).reshape(1, -1)
            xi = mlmnn.transform(xi, Gpredict)
            pred = mlmnn_knns[Gpredict-1].predict(xi)[0]

            xi = getHistogram(rawdata, [ind], K=K, G=G, threshold=threshold)[0][:g+1]
            try:
                assert (xi.flatten() == x[ind].reshape(G, -1)[:g+1].flatten()).all()
            except:
                import pdb
                pdb.set_trace()
            xi = mlmnn.transform(xi.reshape(1, -1), g+1)
            predonGreal = mlmnn_knns[g].predict(xi)[0]

            print 'Gpred={}, Greal={}, Ypred={}, Yreal={}, YpredonGreal={}'.\
                format(Gpredict, g+1, pred, yi, predonGreal)

            if pred == yi: accpred[g] += 1
            if predonGreal == yi: accpredonGreal[g] += 1
        print 'acc on Gpred:', accpred/(count+1)
        print 'acc on Greal:', accpredonGreal/(count+1)
                



