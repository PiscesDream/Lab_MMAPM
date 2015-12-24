import numpy as np
import numpy.linalg as linalg

def LDL(A, combined=False):
    assert(A.shape[0] == A.shape[1])

    L = np.zeros(A.shape)
    D = np.zeros(A.shape)
    n = A.shape[0]

    for i in xrange(n):
        for j in xrange(n):
            if i == j:
                D[i, i] = A[i, i]
                for k in xrange(i):
                    D[i, i] -= (L[i, k] ** 2) * D[k, k]
                L[i, i] = 1
            elif j <= i:
                L[i, j] = A[i, j]
                for k in xrange(j):
                    L[i, j] -= L[i, k] * D[k, k] * L[j, k]
                L[i, j] *= 1/D[j, j]

    if combined:
        return np.dot(L, np.sqrt(D[:,:linalg.matrix_rank(A)]))
    else:
        return L, D

if __name__ == '__main__':
    a = np.array([[1,2,3], [3,2,4]])
    M = np.dot(a.T, a)
    L, D = LDL(M)

    H = np.dot(L, np.sqrt(D[:,:linalg.matrix_rank(M)]))

    print a
    print M
    print L
    print D
    print H
    print np.dot(H, H.T)

    '''
    L =

    1.0000         0         0
    0.8000    1.0000         0
    1.5000    1.2500    1.0000


    D =

    10.0000         0         0
    0    1.6000         0
    0         0    0.0000

    '''
