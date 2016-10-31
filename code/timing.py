
import data
import time
import utils


frame_loader = data.FrameLoader()#, max_videos=2)
frame_loader = utils.ValidationMinibatches(frame_iterator=frame_loader, cache=frame_loader.data_can_fit_in_memory())

for i in range(0, 2):
    start = time.time()
    count = 0
    for data in frame_loader.train:
    #for data in frame_loader:
        count += 1
    print('FrameLoader spent %.4fs' % (time.time() - start))
    print('found %d' % (count))

    start = time.time()
    count = 0
    for data in frame_loader.val:
    #for data in frame_loader:
        count += 1
    print('FrameLoader spent %.4fs' % (time.time() - start))
    print('found %d' % (count))
