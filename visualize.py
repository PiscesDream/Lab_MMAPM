import matplotlib
matplotlib.use('Agg')
from tSNE.draw import visualize
import cPickle

if __name__ == '__main__':
    K = 100 
    Time = 1.0
    M = 10

    trainx, testx, trainy, testy = cPickle.load(open('./features/[K={}][T={}]BoWInGroup.pkl'.format(K, Time),'r'))
    print trainx.shape
    print trainy.shape

    visualize(trainx, trainy, './tSNE/pre-Train.png')
    visualize(testx, testy, './tSNE/pre-Test.png')
   
    from MLMNN import MLMNN
    mlmnn = MLMNN.load('./temp.MLMNN')
    trainx = mlmnn.transform(trainx)
    testx = mlmnn.transform(testx)
    print trainx.shape

    visualize(trainx, trainy, './tSNE/post-Train.png')
    visualize(testx, testy, './tSNE/post-Test.png')



