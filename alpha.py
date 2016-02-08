from MLMNN import MLMNN, loadtest
import numpy as np
import theano
import sys

MODEL_NAME = 'temp.MLMNN' #'models/20160205_rs133.MLMNN'

def loadtest(trainx, testx, trainy, testy, As=None):
    mlmnn = MLMNN.load(MODEL_NAME)

    acc = []
    for Time2 in np.linspace(0.1, 1, 10):
        Time2 = int(Time2*10)
        f = lambda x: x.reshape(x.shape[0], 10, -1)[:, :Time2, :].reshape(x.shape[0], -1)
        ttrainx = f(trainx)
        ttestx = f(testx)
        if As:
            acc.append(mlmnn.fittest(ttrainx, ttestx, trainy, testy, Time2, alpha=As[Time2-1][:Time2]))
        else:
            acc.append(mlmnn.fittest(ttrainx, ttestx, trainy, testy, Time2))

    print acc
    acc = np.array(acc)
    print acc[:,0]
    print acc[:,1]   
    return acc

def svmtest(trainx, testx, trainy, testy):
    mlmnn = MLMNN.load(MODEL_NAME)
    traind = mlmnn.get_DMatrix(trainx, trainx).sum(0)
    testd  = mlmnn.get_DMatrix(testx, trainx).sum(0)
    
    from sklearn import svm
    clf = svm.SVC(kernel='precomputed', C=1.0)
    clf.fit(traind, trainy)
    print clf.predict(traind)
    print trainy
    print (clf.predict(traind)==trainy).sum()

    print clf.predict(testd)
    print testy
    print (clf.predict(testd)==testy).sum()


def testmmcml(trainx, testx, trainy, testy):
    mlmnn = MLMNN.load(MODEL_NAME) # randomstate=32
    traind = mlmnn.get_DMatrix(trainx)

    from MMCML import MMCML
    mmcml = MMCML()
    As = [mmcml.fit(traind[:i], trainy, max_iter=50, lr=5e-5).A.flatten() for i in range(1, 11) ]
    return As

def testlmnn(trainx, testx, trainy, testy):
    mlmnn = MLMNN.load(MODEL_NAME) # randomstate=32
    traind = mlmnn.get_DMatrix(trainx)

    from LMNN_alpha import LMNN_alpha
    lmnn = LMNN_alpha()
    As = [lmnn.fit(traind[:i], trainy, max_iter=50, lr=5e-5).A.flatten() for i in range(1, 11) ]
    return As



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
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.33, random_state=138)
    print trainx.shape
    print trainy.shape
 


    acc1 = loadtest(trainx, testx, trainy, testy)
    As = testlmnn(trainx, testx, trainy, testy)
    acc2 = loadtest(trainx, testx, trainy, testy, As)
    print 'As', As
    print acc1
    print acc2


#   [[ 45.149  17.424]    [[ 45.149  17.424]
#   [ 59.328  40.909]     [ 54.104  39.394]
#   [ 70.896  49.242]     [ 57.836  34.848]
#   [ 80.597  57.576]     [ 75.373  52.273]
#   [ 84.701  70.455]     [ 83.209  61.364]
#   [ 91.045  75.   ]     [ 79.104  65.909]
#   [ 90.672  78.788]     [ 89.552  68.939]
#   [ 91.418  78.03 ]     [ 91.791  75.758]
#   [ 90.672  75.   ]     [ 89.179  75.758]
#   [ 90.672  75.758]]    [ 88.06   77.273]]











