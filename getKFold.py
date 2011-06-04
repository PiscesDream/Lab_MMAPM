from sklearn.cross_validation import KFold
from names import *
import cPickle

if __name__ == '__main__':
    print 'Loading features ...'
#   lf = cPickle.load(open(localFeaturesFilename, 'r'))
#   gf = cPickle.load(open(globalFeaturesFilename, 'r'))
    
    n = 12 # len(lf)
    for train_index, test_index in KFold(n=n, n_folds=4):
#        print "TRAIN: {} TEST: {}".format(train_index, test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

