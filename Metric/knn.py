import numpy as np
from common import load_mnist
from sklearn.decomposition import PCA

def KNN_predict(x, y, predx, L=None, M=None, K=3):
#   if not M:
#       if L:
#           M = np.dot(L, L.T)
#       else:
#           M = np.eye(dim)
    

    pca = PCA(whiten=True)
    pca.fit(x)
#    print pca.explained_variance_ratio_
    components, variance = 0, 0.0
    for components, ele in enumerate(pca.explained_variance_ratio_):
        variance += ele
        if variance > 0.95: break
    components += 1
    print 'n_components=%d'%components
    pca.set_params(n_components=components)
    pca.fit(x)

    x = pca.transform(x)
    predx = pca.transform(predx)

    n, dim = x.shape
    if not L:
        L = np.eye(dim)

    Lx = np.dot(x, L)
    Lpredx = np.dot(predx, L)

    print 'Predicting ...'
    predict = []
    for ind, ele in enumerate(Lpredx):
        dist = ((ele - Lx)**2).sum(1)
        pred = y[dist.argsort()[1:K+1]]
        bincount = np.bincount(pred)
        maxcount = bincount.max()
        candidates = [predy for predy in pred if bincount[predy] == maxcount]
        predict.append( np.random.choice(candidates) )
    return np.array(predict)

if __name__ == '__main__':
    train_data, test_data = load_mnist(percentage=0.1, skip_valid=True)
    train_x, train_y = train_data
    test_x, test_y = test_data

    predict_y = KNN_predict(train_x, train_y, test_x, K=5)
    correct = (predict_y==test_y).sum()
    print 'acc = {}/{} = {}%'.format(correct, 
            predict_y.shape[0],
            float(correct)/predict_y.shape[0]*100
            )

