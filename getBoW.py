import cPickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
import pdb
from names import *

def getCodebook(lf, gf, K): 
    filename = codebookFilename(K)
    if os.path.exists(filename):
        print '{} found'.format(filename)
        return cPickle.load(open(filename, 'r'))
    else:
        lx = []
        gx = []
        for tag in lf:
            lx.extend(lf[tag]['x'])
            gx.extend(gf[tag]['x'])
        lcodebook = _getCodebook(lx, K)
        gcodebook = _getCodebook(gx, K)
        print 'Saving to {} ...'.format(filename)
        cPickle.dump((lcodebook, gcodebook), open(filename, 'w'))
        return lcodebook, gcodebook

def _getCodebook(x, K):
    x = np.array(x).astype('float32')
    x = x.reshape(x.shape[0], -1)
    # normalization
    # x = x/x.sum(1).reshape(x.shape[0], 1)
    x = normalize(x)

    print 'Original shape: {}'.format(x.shape)
    print 'Clustering into {} categories ...'.format(K)

    codebook = KMeans(n_clusters=K, precompute_distances=True, n_jobs=-1)
    codebook.fit(x)
    return codebook



# raw data -> [cluster_index, t]
# data.keys() = ['lx0', 'gx0', ... , 'lxn', 'gxn', 'y']
def encode(K):
    filename = codedFilename(K)
    if os.path.exists(filename):
        print '{} found'.format(filename)
        return dict(np.load(open(filename, 'r')))
    else:
        print 'loading {}'.format(localFeaturesFilename)
        lf = cPickle.load(open(localFeaturesFilename, 'r'))
        print 'loading {}'.format(globalFeaturesFilename)
        gf = cPickle.load(open(globalFeaturesFilename, 'r'))

        lcodebook, gcodebook = getCodebook(lf, gf, K)
        data = _encode(lf, gf, lcodebook, gcodebook)
        print 'Saving to {} ...'.format(filename)
#        np.savez_compress
        np.savez(open(filename, 'w'), **data)
        return data

def _encode(lf, gf, lcodebook, gcodebook):
    assert (lf.keys() == gf.keys())
    data = {}
    y = []

    # differ among datasets
    tagging = lambda x: x.split('_')[-1]
    yset = list(set([tagging(x) for x in lf.keys()]))

    for ind, tag in enumerate(lf):
        y.append(yset.index(tagging(tag)))
        print '{} ==> {}'.format(tag, y[-1])

        lx = lcodebook.predict(normalize(lf[tag]['x']))
        lx = np.vstack([lx, lf[tag]['t']]).astype('float32')
        # 2 row: cluster_index, and timestamp

        gx = gcodebook.predict(normalize(gf[tag]['x']))
        gx = np.vstack([gx, gf[tag]['t']]).astype('float32')
        # 2 row: cluster_index, and timestamp

        data['lx{}'.format(ind)] = lx
        data['gx{}'.format(ind)] = gx
    data['y'] = np.array(y, dtype='int32')
    return data



# coded data -> histogram
# data.keys() = ['x', 'y']
def getHistogram(K, G):
    filename = histogramFilename(K, G)
    if os.path.exists(filename):
        print '{} found'.format(filename)
        data = np.load(open(filename, 'r'))
        return data['x'], data['y']
    else:
        data = encode(K)

        N = (len(data.keys())-1)/2
        print '{} rows will be processed'.format(N)
        x = []
        y = []
        for i in range(N):
            lx = data['lx{}'.format(i)]
            lx = _getHistogram(lx[0], lx[1], K, G)
            # dim = (G, K)

            gx = data['gx{}'.format(i)]
            gx = _getHistogram(gx[0], gx[1], K, G)
            # dim = (G, K)

            x.append(np.hstack([lx, gx]).astype('float32')) # dim=(G, 2*K)
            y.append(data['y'][i])

        print 'Saving to {} ...'.format(filename)
#        np.savez_compress
        np.savez(open(filename, 'w'), x=np.array(x).astype('float32'), y=y)
        return x, y 

def _getHistogram(x, t, K, G, threshold=None):
    tmax = 1.0 #np.max(t)
    splitpoints = np.linspace(0, tmax, G+1)
    splitpoints[-1] = np.inf

    histogram = np.zeros((G, K), dtype='float32')  
    for g, (start, end) in enumerate(zip(splitpoints[:-1], splitpoints[1:])):
        mask = (start<=t) & (t<end)
        if threshold:
            mask = mask & (t < threshold)
        for label in x[mask]:
            histogram[g] += np.bincount([label], minlength=K)
    return histogram

if __name__ == '__main__':
    getHistogram(K=300, G=4)



