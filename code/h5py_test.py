
import h5py
import numpy as np
from sklearn.cross_validation import train_test_split

import data
import utils



frame_loader = data.FrameLoader(shuffle=True)
#frame_loader = utils.Validation(frame_iterator=frame_loader, test_fraction=0.33)
frame_loader = utils.HDF5Validation(frame_iterator=frame_loader, count=frame_loader.inputs.shape[0], test_fraction=0.33)

frame_loader_train = utils.Minibatch(frame_iterator=frame_loader.train)
frame_loader_test  = utils.Minibatch(frame_iterator=frame_loader.test)

import time

for i in range(0, 3):
    count = 0
    start = time.time()
    for inputs, targets in frame_loader_train:
        count += 1
    print('Train count:\t%d\ttime:\t%gs' % (count, time.time() - start))

    count = 0
    start = time.time()
    for inputs, targets in frame_loader_test:
        count += 1
    print('Test count:\t%d\ttime:\t%gs' % (count, time.time() - start))


# inputs, targets = frame_loader.inputs, frame_loader.targets
# frame_count = inputs.shape[0]
#
# indices = np.random.permutation(frame_count)
# n_train = frame_count // 3
# idx_train = indices[0:n_train]
# idx_train = sorted(idx_train.tolist())
# idx_test = indices[n_train:]



#for i, (image, target) in enumerate(frame_loader):
#    if i % 1000 == 0:
#        print(i)
#        print(image.mean())
#        print(image.std())
#        print(image.max())
#
#for frame in frame_loader:
#    print(frame)



import sys
sys.exit()

# Load file
# f = h5py.File("mytestfile.hdf5", "r")
# count = 0
# inputs = f['inputs']
# targets = f['targets']
#
# for i, image in enumerate(inputs):
#     #print(image.shape)
#     #print(targets[i])
#     count += 1
# print(count)
# import sys
# sys.exit()

# Create a new file
f = h5py.File("mytestfile.hdf5", "w")

frame_loader = data.FrameLoaderFromVideo(shuffle=False, found_only=True)
frame_loader = utils.BallPositionPoint(frame_iterator=frame_loader)
frame_loader = utils.ReshapeAndStandardize(frame_iterator=frame_loader)

count = 0
for i, (image, target) in enumerate(frame_loader):
    if i == 0:
        height, width, channels = image.shape
        target_length = len(target)
    count += 1

print('count')
print(count)

#import sys
#sys.exit()

inputs  = f.create_dataset("inputs",  (count, height, width, channels), dtype='float32')
targets = f.create_dataset("targets", (count, target_length), dtype='float32')

for i, (image, target) in enumerate(frame_loader):
    inputs[i,:,:,:] = image
    targets[i,:]    = target

import sys
sys.exit()
