
import data
import time
import utils


MAX_VIDEOS = math.inf
if 'RUNNING_ON_LOCAL' in os.environ:
    MAX_VIDEOS = 4

frame_loader = data.FrameLoader(max_videos=MAX_VIDEOS)
frame_loader = utils.ValidationMinibatches(frame_iterator=frame_loader, cache=frame_loader.data_can_fit_in_memory())

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
