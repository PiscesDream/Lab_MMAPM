import os
import numpy as np
import cPickle
import sys
from subprocess import call

from names import *

MAX_FEATURES = 1000

if __name__ == '__main__':
    # raw format
    # 'tag': { 
    #           'y': label (int, dim=(1,) )
    #           'x': features (numpy array, dim=(feature_count, feature_length))
    #           't': time (int in frame, dim=(feature_count, ) )
    #        }
    
    # global features
    if '-f' in sys.argv or\
        not os.path.exists(globalFeaturesFilename)  or\
        raw_input('%s existed, overwrite?[y/N]'%globalFeaturesFilename)=='y':

        data = {}
        taglist = map(lambda x: x.strip(), open(videolistFile, 'r').readlines() )
        for tag in taglist: 
            print 'handling video {} ...'.format(tag)
            sys.stdout.flush()

            data[tag] = {}
            data[tag]['y'] = int(tag.split('_')[1])
            #command = './tools/dense_trajectory_release_v1.2/release/DenseTrack '+os.path.join('/', tag)+'.avi'
            #print '\texec: {}'.format(command)
            #res = os.popen(command).readlines()
            #with open(os.path.join(denseDir, tag.split('/')[-1])+'.txt', 'w') as f:
            #    f.writelines(res)
            with open(os.path.join(denseDir, tag.split('/')[-1])+'.txt', 'r') as f:
                res = f.readlines()
            res = np.array(map(lambda x: x.strip().split('\t'), res), dtype=float)
            print '\tshape: {}'.format(res.shape)

            data[tag]['x'] = res[:, 10:40]
            data[tag]['t'] = res[:, 9] 

        for tag in data:
            data[tag]['x'] = np.array(data[tag]['x'])
            data[tag]['t'] = np.array(data[tag]['t'])
            l = data[tag]['x'].shape[0]
            if l > MAX_FEATURES:
                index = np.random.choice(range(l), MAX_FEATURES, False)
                data[tag]['x'] = data[tag]['x'][index]
                data[tag]['t'] = data[tag]['t'][index]
            print 'tag[{}] has {} features'.format(tag, data[tag]['x'].shape) 
            sys.stdout.flush()

        print 'Dumping global features to {} ...'.format(globalFeaturesFilename)
        sys.stdout.flush()
        cPickle.dump(data, open(globalFeaturesFilename, 'w'))

    # local features
    if '-f' in sys.argv or \
            not os.path.exists(localFeaturesFilename)  or\
            raw_input('%s existed, overwrite?[y/N]'%localFeaturesFilename)=='y':


        print 'Getting STIP...'
        # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/opencv_alias 
        command = '{}/bin/stipdet.out -i ./BIT/video-list.txt -vpath {} -o {} -det harris3d -vis no'.format(harris3d_dir, '/', POI)
        print command 
        call(command.split(' '))

        print 'Finding STIP...'
        sys.stdout.flush()
        if not os.path.exists(POI):
            raise Exception("Cannot find Point Of Interest file!")

        data = {}
        tag = None
        with open(POI, 'r') as f:
            for line in f:
                if line.strip() == '' or line.startswith('# point-type'):
                    continue
                elif line.startswith('# '):
                    tag = line.strip().split(' ')[-1]
                    data[tag] = {}
                    data[tag]['y'] = int(tag.split('_')[1])
                    data[tag]['x'] = []
                    data[tag]['t'] = []
                else:
                    # point-type y-norm x-norm t-norm y x t sigma2 tau2 dscr-hog(72) dscr-hof(90)
                    data[tag]['x'].append(map(float, line.strip().split(' ')[9:]))
                    data[tag]['t'].append(float(line.strip().split(' ')[3]))

        for tag in data:
            data[tag]['x'] = np.array(data[tag]['x'])
            data[tag]['t'] = np.array(data[tag]['t'])
            l = data[tag]['x'].shape[0]
            if l > MAX_FEATURES:
                index = np.random.choice(range(l), MAX_FEATURES, False)
                data[tag]['x'] = data[tag]['x'][index]
                data[tag]['t'] = data[tag]['t'][index]
            print 'tag[{}] has {} features'.format(tag, data[tag]['x'].shape) 
            sys.stdout.flush()

        print 'Dumping local features to {} ...'.format(localFeaturesFilename)
        sys.stdout.flush()
        cPickle.dump(data, open(localFeaturesFilename, 'w'))



