import os

# unchangable
harris3d_dir = './tools/stip-2.0-linux'

# changable
# featureDir = 'JPL'
dataset_name = 'BIT'
# dataset_name = 'UTI1'
videolistFile = './{}/video-list.txt'.format(dataset_name)
POI = './{}/POI.txt'.format(dataset_name)
denseDir = './{}/dense'.format(dataset_name)
kfolddir = './{}/kfold'.format(dataset_name)
featureDir = './{}/features'.format(dataset_name)
#videoDir = '/home/share/shaofan/BIT-Interaction/videos'

# derived
BoWInGroup100 = os.path.join(featureDir,'[K=100]BoWInGroup.pkl')
BoWInGroup500 = os.path.join(featureDir,'[K=500]BoWInGroup.pkl')
BoWInGroup5 = os.path.join(featureDir,'[K=5]BoWInGroup.pkl')
localFeaturesFilename = os.path.join(featureDir, 'localFeaturesRaw.pkl')
globalFeaturesFilename = os.path.join(featureDir,'globalFeaturesRaw.pkl')
localBoWFilename = os.path.join(featureDir,'localBoW.pkl')
globalBoWFilename = os.path.join(featureDir,'globalBow.pkl')
def globalcodedfilename(K, T): return '{}/[K={}][T={}]BoWInGroup.globalcoded.pkl'.format(featureDir, K, T)

