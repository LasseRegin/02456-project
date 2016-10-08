
import data
import time
import utils


frame_loader = data.FrameLoader(downsample=4, found_only=True)
frame_loader = utils.BallPositionPoint(frame_iterator=frame_loader)
frame_loader = utils.ReshapeAndStandardize(frame_iterator=frame_loader)

frame_loader = utils.Validation(frame_iterator=frame_loader, test_fraction=0.33)

frame_loader_train = utils.Minibatch(frame_iterator=frame_loader.train, batch_size=20)
frame_loader_test  = utils.Minibatch(frame_iterator=frame_loader.test, batch_size=20)

for i in range(0, 2):
    start = time.time()
    count = 0
    for inputs, targets in frame_loader_train:
        count += 1
    print('FrameLoader spent %.4fs' % (time.time() - start))
    print('found %d' % (count))

    start = time.time()
    count = 0
    for inputs, targets in frame_loader_test:
        count += 1
    print('FrameLoader spent %.4fs' % (time.time() - start))
    print('found %d' % (count))
