import sys
import os
sys.path.append('..')
from names import videoDir

if __name__ == '__main__':
    count = 0
    with open('BIT/video-list.txt', 'w') as f:
         for root, dirs, files in os.walk(videoDir, topdown=False):
            for name in files:
                print os.path.join(root, name[:-4])
                count += 1
                f.write('{}\n'.format(os.path.join(root, name[:-4])))
    print('count = {}'.format(count))
