import cPickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
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

    codebook = KMeans(n_clusters=K, precompute_distances=True, n_jobs=-1, **kwargs)
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



def getBagOfWordInGroup(data, K=100, groups=10, T=1.0, **kwargs):
    x = []
    print 'T={}'.format(T)
    for tag in data:
        if data[tag].get('type', 'test') == 'test': 
            continue

        if True: # cluster on all set
            x.extend(data[tag]['x'])
        else:
            x.extend(data[tag]['x'][data[tag]['t']<=T])
#        x.extend(data[tag]['x'])
    x = np.array(x)

    print 'K={}'.format(K)
    print 'All x.shape={}'.format(x.shape)
    codebook = getCodebook(x, K, **kwargs)
    print 'code done'

    x = []
    y = []
    tags = []
    for tag in data:
        curx = []
        cury = []
        for start, end in zip( np.linspace(0, 1, groups+1)[:-1], np.linspace(0, 1, groups+1)[1:]):
            if start >= T: break
            mask = (start<=data[tag]['t']) & (data[tag]['t']<=end)
            curx.append( calcHistogramSum(codebook, data[tag]['x'][mask]) )
            cury.append( data[tag]['y'])
        x.append(curx)
        y.append(cury)
        tags.append(tag)
    x = np.array(x)
    y = np.array(y)
    print 'Final x.shape={}'.format(x.shape)
    print 'Final y.shape={}'.format(y.shape)
    return x, y, tags 

def BIT_DATA_InGroup(lf, gf, **kwargs):
    n = len(lf)
    test_size = 0.33
    test = np.random.choice(range(n), size=(int(n*test_size),), replace=False)
    train = np.array(list(set(range(n)) - set(test)))
    if True or kwargs.get('verbose', False):
        print 'train: {}\n test: {}'.format(train, test)
    for ind, tag in enumerate(lf):
        if ind in train:
            lf[tag]['type'] = gf[tag]['type'] = 'train'
        else:
            lf[tag]['type'] = gf[tag]['type'] = 'test'

    lx, ly, ltags = getBagOfWordInGroup(lf, **kwargs)
    gx, gy, gtags = getBagOfWordInGroup(gf, **kwargs)
    assert(ltags == gtags)
    assert(ly == gy).all()

    tag = map(lambda x: x.split('/')[-2], ltags)
    yset = list(set(tag))
    y = map(lambda x: yset.index(x), tag)

    x = np.concatenate([lx, gx], 2)
    x = x.reshape(x.shape[0], -1)
    y = np.array(y)

    trainx = x[train]
    testx = x[test] 
    trainy = y[train]
    testy = y[test]

    return trainx, testx, trainy, testy
#    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33)
#    return train_x, train_y, test_x, test_y

def SSVM_Data(trainx, testx, trainy, testy, dir):
    with open(os.path.join(dir, 'train.dat'), 'w') as f:
        for x, y in zip(trainx, trainy):
            f.write('{} {}\n'.format(y+1, sparse(x)))

    with open(os.path.join(dir, 'test.dat'), 'w') as f:
        for x, y in zip(testx, testy):
            f.write('{} {}\n'.format(y+1, sparse(x)))
    print '{} data is ready'.format(dir)

def __(K, T):
    trainx, testx, trainy, testy = cPickle.load(open('./features/[K={}][T={}]BoWInGroup.pkl'.format(K, T),'r'))

#    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33)
    dirname = './data/K={}_T={}/'.format(K, T)
    import os
    try:
        os.mkdir(dirname)
    except:
        pass
    SSVM_Data(trainx, testx, trainy, testy, dirname)

def main():
    K = 400
    T = 0.1

    lf = cPickle.load(open(localFeaturesFilename, 'r'))
    gf = cPickle.load(open(globalFeaturesFilename, 'r'))
    
    trainx, testx, trainy, testy = BIT_DATA_InGroup(lf, gf, K=K, verbose=True, T=T)
    cPickle.dump((trainx, testx, trainy, testy), open('./features/[K={}][T={}]BoWInGroup.pkl'.format(K, T), 'w'))

    __(K=K, T=T)
    return
    return 

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

    train_x, train_y, test_x, test_y =  BIT_DATA(lf, gf, K=100, T=1.0, verbose=True)
    print train_x
    print train_y
    print test_x
    print test_y
    SSVM_Data(train_x, train_y, './data')
    print 'done'


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
