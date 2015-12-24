import os

kfolddir = './BIT/kfold'
harris3d_dir = './tools/stip-2.0-linux'
POI = './BIT/POI.txt'
videolistFile = './BIT/video-list.txt'
denseDir = './BIT/dense'
videoDir = '/home/share/shaofan/BIT-Interaction/videos'
featureDir = 'features'
#featureDir = 'JPL'
localFeaturesFilename = os.path.join(featureDir, 'localFeaturesRaw.pkl')
globalFeaturesFilename = os.path.join(featureDir,'globalFeaturesRaw.pkl')
localBoWFilename = os.path.join(featureDir,'localBoW.pkl')
globalBoWFilename = os.path.join(featureDir,'globalBow.pkl')
BoWInGroup100 = os.path.join(featureDir,'[K=100]BoWInGroup.pkl')
BoWInGroup500 = os.path.join(featureDir,'[K=500]BoWInGroup.pkl')
BoWInGroup5 = os.path.join(featureDir,'[K=5]BoWInGroup.pkl')

