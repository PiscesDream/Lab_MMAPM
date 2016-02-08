import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg

class MMCML(object):

    def __init__(self, lmbd=1.0, granularity=10):
        self.lmbd = lmbd
        self.granularity = granularity

    def __Theano_build__(self):
        Td = T.tensor3('Td')
        Ty = T.ivector('Ty')
        Tlr = T.scalar('Tlr')
        #Talpha = T.TensorType(dtype='float32', broadcastable=(0, 1, 1))('alpha')
        A = theano.shared(np.ones((self.D.shape[0]))\
                .astype('float32').reshape(-1, 1, 1), 'A')

        d = (Td * T.addbroadcast(A, 1, 2)).sum(0)
        def eachCost(d, i, y):
            mask = T.eq(y[i], y)
            cost1 = (d * mask).sum()
            cost2 = T.log((T.exp(-d)*mask).sum()  - T.exp(-d[i]))
            return cost1+cost2

        cost, _ = theano.scan(fn=eachCost,
                sequences=[d, T.arange(self.D.shape[1])],
                outputs_info=None,
                non_sequences=[Ty])

        loss = T.sum(cost) 
        grad = T.grad(loss, A)
        newA = A - Tlr*grad #T.maximum(A - Tlr*grad, 0)
        updates = [(A, newA/newA.sum())]

        self.Ttrain = theano.function([Tlr], 
                Tlr*grad,
                givens={Td: self.D, 
                        Ty: self.y},
                updates=updates,
                allow_input_downcast=True,
                on_unused_input='warn')

        self.Tloss = theano.function([], 
                loss,
                givens={Td: self.D, Ty: self.y},
                allow_input_downcast=True)


#       eig, eigv = linalg.eig((d+d.T)/2.0)
#       self.Tmineig = theano.function([], 
#               T.min(eig),
#               givens={Td: self.D},
#               allow_input_downcast=True)

        self.Tmindist = theano.function([], 
                T.min(d),
                givens={Td: self.D},
                allow_input_downcast=True)




        self.Ttransform = theano.function([Td], (Td*T.addbroadcast(A, 1, 2)).sum(0), allow_input_downcast=True)
        self.TA = A

    def fit(self, D, y, max_iter=1000, lr=1e-3):
        self.D = np.array(D, dtype=theano.config.floatX)
        self.y = np.array(y, dtype='int32')
        self.classes = len(set(y))
        self.__Theano_build__()
        
        from datetime import datetime
        print 'Initial loss: {}'.format(self.Tloss())
        last_loss = np.inf
        for i in range(max_iter):
            lastA = self.A
            #startt = datetime.now()
            grad = self.Ttrain(lr).flatten()
#           mineig = self.Tmineig()
            mindist = self.Tmindist()
            #print 'Training use:', (datetime.now() - startt).total_seconds()
            loss = self.Tloss()
            print 'Iter[{}]: A={} lr={}, L(A)={} mindist={}'.\
                    format(i, self.A.flatten(), lr, loss, mindist, )
#           print 'D:', self.transform(self.D)
            if loss >= last_loss or np.isnan(loss) or mindist < 0 or np.min(self.A) < 0:
                lr *= 0.5 
                self.A = lastA # backtrack lr and A
            else:
                lr *= 1.01
                last_loss = loss
        return self
        
    def transform(self, X):
        return self.Ttransform(X)

    @property
    def A(self):
        return self.TA.get_value()

    @A.setter
    def A(self, value):
        self.TA.set_value(value)


def getD(x):
    n, m = x.shape
    d = np.zeros((m, n, n))
    for i, xi in enumerate(x):
        for j, xj in enumerate(x):
            d[:, i, j] = (xi - xj)**2
    return d


if __name__ == '__main__':
    np.set_printoptions(linewidth=100)
    print "Test 1, simple test"
    X = np.array([[10, 0, 0], [20, 1, 1], [20, 2, 0], [10, 3, 1]], dtype='float32')
    D = getD(X)
    y = np.array([0, 0, 1, 1], dtype='int32')
    mcml = MMCML()
    mcml.fit(D, y, max_iter=1000, lr=5e-4)
#   print mcml.A
#   print mcml.transform(X)


#   import sys; sys.path.append('/home/shaofan/Projects') 
#   from FastML import KNN
#   print "\nTest 3, LFW"
#   from sklearn.datasets import fetch_lfw_people
#   from sklearn.decomposition import PCA 
#   from sklearn.cross_validation import train_test_split
#   from sklearn.preprocessing import normalize
#   lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
#   pca = PCA(n_components=10)
#   X = pca.fit_transform(lfw_people.data)
#   X = normalize(X)
#   y = lfw_people.target
#   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
#   knn = KNN(n_neighbors=2)
#   pred = knn.fit(X_train, y_train).predict(X_test)
#   print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )

#   knn = KNN(n_neighbors=2)
#   mcml = MCML()
#   for _ in range(100):
#       X_train = mcml.fit(X_train, y_train, max_iter=5, lr=1e-6).transform(X_train)
#       X_test = mcml.transform(X_test)

#       pred = knn.fit(X_train, y_train).predict(X_test)
#       print '{}/{}'.format( (pred == y_test).sum(), len(y_test) )


