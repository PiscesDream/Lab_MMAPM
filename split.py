import cPickle
import numpy as np
import sys
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

if __name__ == '__main__':
    K = 200
    Time = 1.0
    groups = 10
    from names import featureDir, globalcodedfilename
    if groups == 10:
        x, y = cPickle.load(open('{}/[K={}][T={}]BoWInGroup.pkl'.format(featureDir, K, 1.0), 'r'))
    else:
        x, y = cPickle.load(open(globalcodedfilename(K, 1.0, groups), 'r'))

    newx = []
    newy = []
    for xi, yi in zip(x, y):
        xi = xi.reshape(10, -1)
        yi = np.array([yi * 10] * 10) + np.arange(10)
        newx.append(xi)
        newy.append(yi)
    x = np.concatenate(newx)
    y = np.concatenate(newy)
    y[y<40] = 0
    y[y!=0] = 1

    pca = PCA(n_components=100)
    x = pca.fit_transform(x)
#    x = normalize(x)
    print sum(pca.explained_variance_ratio_)

    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.10) 
    print trainx.shape
    print trainy.shape
    print len(set(trainy))

    from sklearn.svm import SVC 
#    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import classification_report 
#    svm = OneVsRestClassifier(SVC(C=0.11)) # 0.90
    svm = SVC(C=100.11) # 0.90
    svm.fit(trainx, trainy)
    print classification_report(testy, svm.predict(testx))

    import sys; sys.path.append('/home/shaofan/Projects') 
    from FastML import NCA, KNN, MCML
    knn = KNN(n_neighbors=8)
    knn.fit(trainx, trainy)
    print classification_report(testy, knn.predict(testx))

#   nca = NCA()
#   for i in range(100):
#       trainx = nca.fit(trainx, trainy, max_iter=5, lr=5e-3).transform(trainx)
#       testx = nca.transform(testx)
#      
#       print classification_report(testy, knn.predict(testx))
#       sys.stdout.flush()

