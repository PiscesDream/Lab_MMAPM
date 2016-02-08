import numpy as np
import theano
import theano.tensor as T
import theano.tensor.nlinalg as linalg
from datetime import datetime
import sys

def colize(x):
    return [x[:, i] for i in range(x.shape[1])]

class LMNN_alpha(object):

    def __init__(self, lmbd=1.0, granularity=10, K=3, tripleCount=1000):
        self.lmbd = lmbd
        self.granularity = granularity
        self.K = K
        self.tripleCount = tripleCount


    def __Theano_build__(self):
        Td = T.tensor3('Td')
        Ty = T.ivector('Ty')
        Tlr = T.scalar('Tlr')
        #Talpha = T.TensorType(dtype='float32', broadcastable=(0, 1, 1))('alpha')
        A = theano.shared(np.ones((self.D.shape[0]))\
                .astype('float32').reshape(-1, 1, 1), 'A')
        Ttriple   = [T.ivector('triple'+x) for x in ['i', 'j', 'l']]
        Tneighbor = [T.ivector('neighbor'+x) for x in ['i', 'j']]

        d = (Td * T.addbroadcast(A, 1, 2)).sum(0)
        pull_error, _ = theano.scan(
            fn = lambda i, j, d: d[i, j],
            sequences=[Tneighbor[0], Tneighbor[1]],
            outputs_info=None,
            non_sequences=[d])
        pull_error = pull_error.sum()

        push_error, _ = theano.scan(
            fn = lambda i, j, l, d: T.neq(Ty[i], Ty[l]) * T.maximum((d[i]-d[j]) - (d[i]-d[l]) +1, 0),
            sequences=[Ttriple[0], Ttriple[1], Ttriple[2]],
            outputs_info=None,
            non_sequences=[d])
#       zerocount = T.eq(linalg.diag(mask*T.maximum(lossij - lossil + 1, 0)), 0).sum()

        error = pull_error.sum() + push_error.sum()
        grad = T.grad(error, A)
        newA = A - Tlr*grad #T.maximum(A - Tlr*grad, 0)
        updates = [(A, newA/newA.sum())]

        self.Ttrain = theano.function(Ttriple+Tneighbor+[Tlr], 
                Tlr*grad,
                givens={Td: self.D, 
                        Ty: self.y},
                updates=updates,
                allow_input_downcast=True,
                on_unused_input='warn')

        self.Tloss = theano.function(Ttriple+Tneighbor, 
                error,
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
        
        neighbors = self.get_neighbors(D.sum(0), y)
        active_set = self.get_active_set(D.sum(0), y, neighbors)
        print 'Initial loss: {}'.format(self.Tloss(*(colize(active_set)+colize(neighbors)) ) )
        last_loss = np.inf
        for i in range(max_iter):
            lastA = self.A
            #startt = datetime.now()
            grad = self.Ttrain(*(colize(active_set)+colize(neighbors)+[lr])).flatten()
#           mineig = self.Tmineig()
            mindist = self.Tmindist()
            #print 'Training use:', (datetime.now() - startt).total_seconds()
            loss = self.Tloss(*(colize(active_set)+colize(neighbors)))
            print 'Iter[{}]: A={} lr={}, L(A)={} mindist={}'.\
                    format(i, self.A.flatten(), lr, loss, mindist, )
#           print 'D:', self.transform(self.D)
            if loss >= last_loss or np.isnan(loss) or mindist < 0 or np.min(self.A) < 0:
                lr *= 0.5 
                self.A = lastA # backtrack lr and A
            else:
                lr *= 1.01
                last_loss = loss
            sys.stdout.flush()
        return self
        
    def transform(self, X):
        return self.Ttransform(X)

    @property
    def A(self):
        return self.TA.get_value()

    @A.setter
    def A(self, value):
        self.TA.set_value(value)

    def get_active_set(self, d, y, neighbors):
        result = []
        ijcandidates = neighbors[np.random.choice(range(neighbors.shape[0]), size=(self.tripleCount))]
        lcandidates = np.random.randint(0, d.shape[0], size=(self.tripleCount, 1) )
        ijl = np.hstack([ijcandidates, lcandidates]).astype('int32')
        return ijl 

    def get_neighbors(self, d, y):
        # shared neighbor
        n = d.shape[0]
        d = d.reshape(n, -1)
        neighbors = np.zeros((n, self.K), dtype=int)
        yset = np.unique(y)
        COST = 0.0
        for c in yset:
            mask = y==c
            ind = np.arange(n)[mask]
            _d = d[mask]

            for i in xrange(n):
                if y[i] != c: continue
                cost = d[i][mask]
                neighbors[i] = ind[cost.argsort()[1:self.K+1]]
        #       COST += sum(sorted(cost)[1:self.K+1])
        #if self.verbose: print 'neighbor cost: {}'.format(COST)
        #self.neighbors = neighbors

        # repeat an arange array, stack, and transpose
        #self.neighborpairs = np.array(neighborpairs, dtype='int32')
        res = np.vstack([np.repeat(np.arange(d.shape[0]), self.K), neighbors.flatten()]).T.astype('int32')
        return res 





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
    print D
    y = np.array([0, 0, 1, 1], dtype='int32')
    lmnn = LMNN_alpha()
    lmnn.fit(D, y, max_iter=1000, lr=5e-4)
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


