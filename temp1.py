import theano
import theano.tensor as T
import numpy as np

if __name__ == '__main__':
    a = theano.shared(value=np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5]).\
            reshape(1,-1,1).astype('float32'))
    x = T.tensor3('x')

    f = theano.function([x], x * T.addbroadcast(a, 2))
    data = np.random.randint(0, 100, size=(1, 5, 3)).astype('float32')
