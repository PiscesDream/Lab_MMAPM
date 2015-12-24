import numpy as np
import numpy.linalg as linalg
from common import load_mnist, knn
from knn import KNN_predict
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from metric_learn import LMNN as _LMNN
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split

# from metric_learn import LMNN

def selfouter(x):
    return np.outer(x, x)

class LMNN(object):
    def __init__(self, K=3, mu=0.5, maxS=500):
        self.K = K
        self.mu = mu
        self.maxS = maxS
        pass

    def __init_gradient(self):
#        if self.verbose: print 'Calculating initial gradient ...'
        neighbors = np.zeros((self.n, self.K), dtype=int)
        yset = set(self.y)
        for c in yset:
            mask = self.y==c
            ind = np.arange(self.n)[mask]
            x = self.x[mask]

            for i in xrange(self.n):
                if self.y[i] != c: continue
                neighbors[i] = ind[((self.x[i] - x)**2).sum(1).argsort()[1:self.K+1]]
        self.neighbors = neighbors

        G0 = np.zeros_like(self.M)
        for i, neighbor in enumerate(self.neighbors): 
            for j in neighbor:
                G0 += selfouter(self.x[i]-self.x[j])
        return (1-self.mu) * G0 

    def __error(self, violate_set):
        pull_error = 0.
        for i, ele in enumerate(self.neighbors):
            for j in ele:
                pull_error += (1-self.mu) * np.trace(self.M.dot(selfouter(self.x[i]-self.x[j])))

        push_error = 0.
        for i, j, l in violate_set:
            push_error += self.mu * \
                    max(0., \
                        1\
                        + np.trace(self.M.dot(selfouter(self.x[i]-self.x[j]))) \
                        - np.trace(self.M.dot(selfouter(self.x[i]-self.x[l]))) )

        return pull_error, push_error


    def __get_violate_set(self, active_set=None):
        if not active_set:
            result = []
            if not self.maxS:
                for i in range(self.n):
                    for l in range(self.n):
                        if self.y[i] == self.y[l]: continue
                        for j in self.neighbors[i]:
                            vij = selfouter(self.x[i]-self.x[j])
                            vil = selfouter(self.x[i]-self.x[l])
                            loss = 1+\
                                    np.trace(self.M.dot(vij)) - \
                                    np.trace(self.M.dot(vil)) 
                            if loss > 0:
                                result.append((i, j, l)) #vij, vil ))
            else:
                candidates = np.random.randint(0, self.n, size=(self.maxS, 2))
                jcandidates = np.random.randint(0, self.K, size=(self.maxS) )
                for (i, l), j in zip(candidates, jcandidates):
#                while len(result) < self.maxS:
#                    i, l = np.random.randint(0, self.n, size=(2,))
#                    j = np.random.choice(self.neighbors[i])
                    if self.y[i] == self.y[l]: continue

                    vij = selfouter(self.x[i]-self.x[j])
                    vil = selfouter(self.x[i]-self.x[l])
                    loss = 1+\
                            np.trace(self.M.dot(vij)) - \
                            np.trace(self.M.dot(vil)) 
                    if loss > 0:
                        result.append((i, j, l)) #vij, vil ))
            return result

        if self.maxS and len(active_set) > self.maxS:
            np.random.shuffle(active_set)
            active_set = active_set[:self.maxS]

        result = []
        for i, j, l in active_set:
#            if self.verbose: print '\t\tCalc ({}, {}, {})'.format(i, j, l)
            vij = selfouter(self.x[i]-self.x[j])
            vil = selfouter(self.x[i]-self.x[l])
            loss = 1+\
                    np.trace(self.M.dot(vij)) - \
                    np.trace(self.M.dot(vil)) 
            if loss > 0:
                result.append((i, j, l)) #vij, vil ))

        return result

    def __calc_diff(self, setA, setB):
        diff = np.zeros_like(self.M)
        for i, j, l in setA:
            if (i, j, l) in setB: continue
            diff += selfouter(self.x[i] - self.x[j]) - \
                    selfouter(self.x[i] - self.x[l])
        return diff

    def __project_sd(self, mat):
#        if self.verbose: print '\tProjecting M ...'
        eig, eigv = linalg.eig(mat)
        eig = np.diag(eig)
        eig[eig < 0] = 0
        return eigv.dot(eig).dot(linalg.inv(eigv))

    def fit(self, x, y, verbose=False, lr=1e-7, max_iter=100, gradient_cycle=20, tol=1e-20):
        self.x = x
        self.y = y
        self.n, self.m = x.shape
        self.verbose = verbose
        last_error = np.inf, np.inf

        self.M = np.eye(self.m)
        t = 0
        active_set, violate_set = [], []
        gradient = self.__init_gradient()
        while t < max_iter:
            if t % gradient_cycle == 0:
                new_violate_set = self.__get_violate_set()
                new_active_set = new_violate_set#+active_set 
            else:
                new_violate_set = self.__get_violate_set(active_set)
                new_active_set = active_set
            
            new_gradient = gradient\
                    - self.mu * self.__calc_diff(violate_set, new_violate_set)\
                    + self.mu * self.__calc_diff(new_violate_set, violate_set)
            newM = self.__project_sd(self.M - lr * new_gradient)

            error = self.__error(new_active_set)
            diff = ((newM-self.M)**2).sum()
            if verbose: print 'Iter={}, error={}, diff={} lr={} Violate_S={}'.format(t, error, diff, lr, len(new_violate_set))
            lr = lr * 1.01 if sum(error) < sum(last_error) else lr * 0.50
            if diff < tol: break

            last_error = error
            active_set = new_active_set
            violate_set = new_violate_set
            gradient = new_gradient
            self.M = newM
            t += 1

        return self


if __name__ == '__main__':
#   iris_data = load_iris()
#   X = iris_data['data']
#   Y = iris_data['target']
#    trainx, testx, trainy, testy = train_test_split(X, Y, test_size=0.66)

    (trainx, trainy), (testx, testy) = load_mnist(percentage=0.01, skip_valid=True)

    pca = PCA(whiten=True)
    pca.fit(trainx)
    components, variance = 0, 0.0
    for components, ele in enumerate(pca.explained_variance_ratio_):
        variance += ele
        if variance > 0.90: break
    components += 1
    print 'n_components=%d'%components
    pca.set_params(n_components=components)
    pca.fit(trainx)

    trainx = pca.transform(trainx)
    testx = pca.transform(testx)

    K = 3
    dim = components
    knn(trainx, testx, trainy, testy, np.eye(dim), K)
    knn(trainx, testx, trainy, testy, np.random.rand(dim, dim), K)

    lmnn = LMNN(K=K, mu=0.5)
    lmnn.fit(trainx, trainy, lr=2e-6, max_iter=1500, gradient_cycle=100, verbose=True)
    knn(trainx, testx, trainy, testy, lmnn.M, K)

    lmnn = _LMNN(k=K, learn_rate=2e-6, max_iter=1000)
    L = lmnn.fit(trainx, trainy, verbose=True).L
    M = L.dot(L)
    knn(trainx, testx, trainy, testy, M, K)
