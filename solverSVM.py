import cPickle
import numpy as np
import sys
from names import *
from sklearn.cross_validation import train_test_split
from pdb import set_trace

def aggregate(lf, gf, T=1.0):
    train_x = []
    train_y = []

    test_x = []
    test_y = []
    for ele in lf:
        # print lf[ele]['BoW'][lf[ele]['t'] < T].shape
        # print gf[ele]['BoW'][gf[ele]['t'] < T].shape
        assert (lf[ele]['y'] == gf[ele]['y'])
        if lf[ele]['type'] == 'train':
            train_x.append( np.array([lf[ele]['BoW'][lf[ele]['t'] <= T].sum(0), gf[ele]['BoW'][gf[ele]['t'] < T].sum(0)]).flatten() )
            train_y.append( lf[ele]['y'] )
        else:
            test_x.append( np.array([lf[ele]['BoW'][lf[ele]['t'] <= T].sum(0), gf[ele]['BoW'][gf[ele]['t'] < T].sum(0)]).flatten() )
            test_y.append( lf[ele]['y'] )

    train_x = np.array(train_x)
    train_y = np.array(train_y, dtype=int)
    test_x = np.array(test_x)
    test_y = np.array(test_y, dtype=int)

    return train_x, test_x, train_y, test_y

if __name__ == '__main__':
    from sklearn import svm
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from kernels import chi_square_kernel, histogram_intersection_kernel, zero_kernel, multichannel_wrapper
    from getBoW import BIT_DATA 

    print 'Loading features ...'
    lf = cPickle.load(open(localFeaturesFilename, 'r'))
    gf = cPickle.load(open(globalFeaturesFilename, 'r'))

#       from file
#       lf = cPickle.load(open(localBoWFilename , 'r'))
#       gf = cPickle.load(open(globalBoWFilename , 'r'))

    print("Training SVM")
    TIMES = 10
    l = []
#    for i in range(TIMES): 
#        percentage = 0.1
    np.set_printoptions(linewidth=200)
    for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        print 'percentage = %.2f%%' % (percentage*100)
        l = []
        for i in range(TIMES):
            train_x, train_y, test_x, test_y = BIT_DATA(lf, gf, K=500, T=percentage, slient=True)
#            print 'Train distribution:', np.bincount(train_y.flatten())/float(train_y.shape[0])
#            print 'Test distribution:', np.bincount(test_y.flatten())/float(test_y.shape[0])

            train_y = label_binarize(train_y-1, classes=range(7))
            test_y = label_binarize(test_y-1, classes=range(7))

            #
            classifier = OneVsRestClassifier(svm.SVC(kernel=multichannel_wrapper(2, chi_square_kernel), probability=True))
            #classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
            y_score = classifier.fit(train_x, train_y).decision_function(test_x)
            l.append(float((test_y.argmax(1) == y_score.argmax(1)).sum())/y_score.shape[0]*100)

            print '\taccuracy = %.3f%%'%l[-1]
            sys.stdout.flush()
        print '>>> average accuracy = %.3f%%'%np.mean(l)


