import cPickle
import numpy as np
from sklearn.cluster import KMeans
import pdb

from names import *

def getCodebook(x, K, **kwargs):
    x = np.array(x) 
    x = x.reshape(x.shape[0], -1)
    # normalization
    x = x/x.sum(1).reshape(x.shape[0], 1)

    if kwargs.get('verbose', False):
        print('Original shape:', x.shape)
        print('Clustering into %d categories ...'%K)

    codebook = KMeans(n_clusters=K, precompute_distances=True, n_jobs=4, **kwargs)
    codebook.fit(x)
    return codebook 

def calcHistogramEach(codebook, x):
    x = np.array(x)
    if x.shape[0] == 0: return []
    x = x.reshape(x.shape[0], -1)
    # normalize
    x = x/x.sum(1).reshape(x.shape[0], 1)
    
    labels = codebook.predict(x)
    histogram = np.array([np.bincount([label], minlength=codebook.n_clusters) for label in labels])
    return histogram

def calcHistogramSum(codebook, x):
    x = np.array(x)
    if x.shape[0] == 0: return np.zeros((codebook.n_clusters, ))
    x = x.reshape(x.shape[0], -1)
    # normalize
    x = x/x.sum(1).reshape(x.shape[0], 1)
    
    labels = codebook.predict(x)
    acc = np.zeros((codebook.n_clusters, ))
    for label in labels:
        acc += np.bincount([label], minlength=codebook.n_clusters)
    return acc 

def getBagOfWord(data, K, given_range, query_range, **kwargs):
    x = []
    for tag in data:
        if data[tag].get('type', 'train') == 'train':
            x.extend(data[tag]['x'][(given_range[0]<=data[tag]['t']) & (data[tag]['t']<=given_range[1])])

    codebook = getCodebook(x, K, **kwargs)
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    tags = []
    for tag in data:
        mask = (query_range[0]<=data[tag]['t']) & (data[tag]['t']<=query_range[1])
        if data[tag].get('type', 'train') == 'train':
            train_x.append( calcHistogramSum(codebook, data[tag]['x'][mask]) )
            train_y.append( data[tag]['y'])
        else:
            test_x.append( calcHistogramSum(codebook, data[tag]['x'][mask]) )
            test_y.append( data[tag]['y'])
        tags.append(tag)

        # save space
        # del data[tag]['x']
    return train_x, train_y, test_x, test_y, tags

from sklearn.cross_validation import train_test_split
def BIT_DATA(lf, gf, K=400, T=1.0, **kwargs):
    ltrain_x, ltrain_y, ltest_x, ltest_y, ltags = getBagOfWord(lf, K, [0, T], [0, T], **kwargs)
    gtrain_x, gtrain_y, gtest_x, gtest_y, gtags = getBagOfWord(gf, K, [0, T], [0, T], **kwargs)
    assert(ltags == gtags)
    assert(ltrain_y == gtrain_y)
    assert(ltest_y == gtest_y)

    train_x = np.concatenate([ltrain_x, gtrain_x], 1)
    train_y = np.array(ltrain_y)
    test_x = np.concatenate([ltest_x, gtest_x], 1)
    test_y = np.array(ltest_y)
    #print test_x

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.33)
    return train_x, train_y, test_x, test_y



def JPL_DATA(lf, gf, K=400, T=1.0, **kwargs):
    train = set(np.random.choice(range(1, 13), size=(6,), replace=False))
    test = set(range(1, 13)) - train
    if kwargs.get('verbose', False):
        print 'train: {}, test: {}'.format(train, test)

    for tag in lf:
        if int(tag.split('_')[0]) in train:
            lf[tag]['type'] = 'train'
            gf[tag]['type'] = 'train'
        else:
            lf[tag]['type'] = 'test'
            gf[tag]['type'] = 'test'

    ltrain_x, ltrain_y, ltest_x, ltest_y, ltags = getBagOfWord(lf, K, [0, T], [0, T], **kwargs)
    gtrain_x, gtrain_y, gtest_x, gtest_y, gtags = getBagOfWord(gf, K, [0, T], [0, T], **kwargs)
    assert(ltags == gtags)
    assert(ltrain_y == gtrain_y)
    assert(ltest_y == gtest_y)
    train_x = np.concatenate([ltrain_x, gtrain_x], 1)
    train_y = np.array(ltrain_y)
    test_x = np.concatenate([ltest_x, gtest_x], 1)
    test_y = np.array(ltest_y)
    return train_x, train_y, test_x, test_y

## feature for MMAPM
def g(lf, gf, given_range, query_range, K=5, **kwargs):
    ltrain_x, ltrain_y, ltest_x, ltest_y, ltags = getBagOfWord(lf, K, given_range, query_range, **kwargs)
    gtrain_x, gtrain_y, gtest_x, gtest_y, gtags = getBagOfWord(gf, K, given_range, query_range, **kwargs)

    assert(ltags == gtags)
    assert(ltrain_y == gtrain_y)
    assert(ltest_y == gtest_y)

    train_x = np.concatenate([ltrain_x, gtrain_x], 1)
    test_x = np.concatenate([ltest_x, gtest_x], 1)

    return train_x, ltrain_y, test_x, ltest_y 

def sparse(x):
    return ' '.join(map(lambda ele: '{}:{}'.format(ele[0], ele[1]),
        zip(range(1, len(x)+1), x) ) )

def main():
    lf = cPickle.load(open(localFeaturesFilename, 'r'))
    gf = cPickle.load(open(globalFeaturesFilename, 'r'))

#   for tag in lf:
#       print '{} -> {}'.format(lf[tag]['y'], tag.split('/')[-2])
#       lf[tag]['y'] = tag.split('/')[-2]
#   for tag in gf:
#       print '{} -> {}'.format(gf[tag]['y'], tag.split('/')[-2])
#       gf[tag]['y'] = tag.split('/')[-2]

#   l = []
#   for tag in lf:
#       if lf[tag]['y'] not in l:
#           l.append(lf[tag]['y'])
#   for tag in lf:
#       print '{} -> {}'.format(lf[tag]['y'], l.index(lf[tag]['y']))
#       lf[tag]['y'] = l.index(lf[tag]['y'])
#   for tag in gf:
#       print '{} -> {}'.format(gf[tag]['y'], l.index(gf[tag]['y']))
#       gf[tag]['y'] = l.index(gf[tag]['y'])

#   cPickle.dump(lf, open(localFeaturesFilename, 'w'))
#   cPickle.dump(gf, open(globalFeaturesFilename, 'w'))
#   return

    train_x, train_y, test_x, test_y =  BIT_DATA(lf, gf, K=3, T=1.0, verbose=True)
    print train_x
    print train_y
    print test_x
    print test_y
    with open('data/train.dat', 'w') as f:
        for x, y in zip(train_x, train_y):
            f.write('{} {}\n'.format(y+1, sparse(x)))

    with open('data/test.dat', 'w') as f:
        for x, y in zip(test_x, test_y):
            f.write('{} {}\n'.format(y+1, sparse(x)))


    # print fpa_data(lf, gf)

    # cpickle.dump(lf, open(localbowfilename , 'w'))
    # cpickle.dump(gf, open(globalbowfilename , 'w'))


    # checking
#   print [lf[ele]['type'] for ele in lf]
#   print [lf[ele]['bow'].shape for ele in lf]
#   print [lf[ele]['t'].shape for ele in lf]
#   print [gf[ele]['type'] for ele in gf]
#   print [gf[ele]['bow'].shape for ele in gf]
#   print [gf[ele]['t'].shape for ele in lf]
#   assert [lf[ele]['type'] for ele in lf] == [gf[ele]['type'] for ele in gf]    


if __name__ == '__main__':
    main()
