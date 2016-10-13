
import data
import utils

# Training parameters
NUM_EPOCHS = 50
LEARNING_RATE = 1e-2

# Intialize frame loader
frame_loader = data.FrameLoaderFromVideo()

count = 0
for frame in frame_loader:
    print(frame.image.shape)
    count += 1
    if count % 1000 == 0:
        print(count)
print('Count: %d' % (count))
#frame_loader = utils.BallPositionPoint(frame_iterator=frame_loader)
#frame_loader = utils.ReshapeAndStandardize(frame_iterator=frame_loader)
