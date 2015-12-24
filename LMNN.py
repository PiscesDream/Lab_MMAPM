import numpy as np
import cPickle
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric  
from sklearn.cross_validation import train_test_split 

from metric_learn import LMNN as _LMNN
from Metric.LMNN import LMNN
from Metric.LMNN2 import LMNN as LMNN_GPU
from Metric.common import knn

if __name__ == '__main__':
    K = 100 
    T = 1.0
#    x, y = cPickle.load(open(r'./features/[K={}][T={}]BoWInGroup.pkl'.format(K, T),'r'))

#    x = x.reshape(x.shape[0], 10, -1).sum(1)

#    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.20)
    trainx, testx, trainy, testy = cPickle.load(open('./features/[K={}][T={}]BoWInGroup.pkl'.format(K, T),'r'))
    # accumulation
#   trainx = trainx.reshape(trainx.shape[0], 10, -1).sum(1)
#   testx = testx.reshape(testx.shape[0], 10, -1).sum(1)
    components = trainx.shape[1]
    print trainx.shape
    print trainy.shape

    pca = PCA(whiten=True)
    pca.fit(trainx)
    components, variance = 0, 0.0
    for components, ele in enumerate(pca.explained_variance_ratio_):
        variance += ele
        if variance > 0.95: break
    components += 1
    print 'n_components=%d'%components
    pca.set_params(n_components=components)
    pca.fit(trainx)

    trainx = pca.transform(trainx)
    testx = pca.transform(testx)

    K = 9 
    dim = components
#   knn(trainx, testx, trainy, testy, np.eye(dim), K)
#   knn(trainx, testx, trainy, testy, np.random.rand(dim, dim), K)

    # My LMNN
    lmnn = LMNN_GPU(K=K, mu=0.5, maxS=10000, dim=dim)
    print 'pre-train-acc: %.3f%% pre-test-acc: %.3f%% %s'%(knn(trainx, trainx, trainy, trainy, lmnn._M, K), knn(trainx, testx, trainy, testy, lmnn._M, K), ' '*30)
    lmnn.fit(trainx, trainy, lr=5e-6, max_iter=1000, reset=50, verbose=True)
    print 'pre-train-acc: %.3f%% pre-test-acc: %.3f%% %s'%(knn(trainx, trainx, trainy, trainy, lmnn._M, K), knn(trainx, testx, trainy, testy, lmnn._M, K), ' '*30)
    #knn(trainx, testx, trainy, testy, lmnn.M, K)
#   L = lmnn.L
#   knn(trainx.dot(L), testx.dot(L), trainy, testy, np.eye(L.shape[1]), K)


#   # metric learn
#   lmnn = _LMNN(k=K, learn_rate=1e-5, max_iter=200) 
#   L = lmnn.fit(trainx, trainy, verbose=True).L
#   knn(trainx, testx, trainy, testy, L.dot(L), K)

