import os

# unchangable
harris3d_dir = './tools/stip-2.0-linux'

# changable
# featureDir = 'JPL'
dataset_name = 'BIT'
#dataset_name = 'UTI'
#dataset_name = 'TV'
#dataset_name = 'UCF11'
videolistFile = './{}/video-list.txt'.format(dataset_name)
POI = './{}/POI.txt'.format(dataset_name)
denseDir = './{}/dense'.format(dataset_name)
kfolddir = './{}/kfold'.format(dataset_name)
featureDir = './{}/features'.format(dataset_name)
#videoDir = '/home/share/shaofan/BIT-Interaction/videos'

# derived
localFeaturesFilename = os.path.join(featureDir, 'localFeaturesRaw.pkl')
globalFeaturesFilename = os.path.join(featureDir,'globalFeaturesRaw.pkl')
localBoWFilename = os.path.join(featureDir,'localBoW.pkl')
globalBoWFilename = os.path.join(featureDir,'globalBow.pkl')

def globalcodedfilename(K, T, groups=10): 
    if groups == 10:
        return '{}/[K={}][T={}]BoWInGroup.globalcoded.pkl'.format(featureDir, K, T)
    else:
        return '{}/[K={}][T={}][G={}]BoWInGroup.globalcoded.pkl'.format(featureDir, K, T, groups)

def codebookFilename(K):
    return os.path.abspath('{}/[K={}]Codebook.pkl'.format(featureDir, K))

def codedFilename(K):
    return os.path.abspath('{}/[K={}]Coded.npz'.format(featureDir, K))

def histogramFilename(K, G): 
    return os.path.abspath('{}/[K={}][G={}]Histogram.npz'.format(featureDir, K, G))



