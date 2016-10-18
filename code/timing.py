
import data
import time
import utils


frame_loader = data.FrameLoader(shuffle=True)
frame_loader = utils.ValidationMinibatches(frame_iterator=frame_loader)

for i in range(0, 2):
    start = time.time()
    count = 0
    for inputs, targets in frame_loader.train:
        count += 1
    print('FrameLoader spent %.4fs' % (time.time() - start))
    print('found %d' % (count))

    start = time.time()
    count = 0
    for inputs, targets in frame_loader.val:
        count += 1
    print('FrameLoader spent %.4fs' % (time.time() - start))
    print('found %d' % (count))
