
import os
import data
import utils
import network

import numpy as np
import tensorflow as tf

# Intialize frame loader
frame_loader = data.FrameLoader(shuffle=True, validation_group='test')
height, width = frame_loader.data.target_height, frame_loader.data.target_width

frame_loader = utils.MinibatchUncached(frame_iterator=frame_loader)

# Setup network
nn = network.SimpleRegressor(name='simple-model-1',
                             input_shape=(None, height, width),
                             target_shape=(None, 2),
                             verbose=True)

with tf.Session() as sess:
    nn.load(sess)

    test_loss = 0.
    test_batches = 0
    for images, targets in frame_loader:
        images /= images.std() # TODO: do somewhere else
        test_loss += nn.val_op(session=sess, x=images, y=targets)
        predictions = nn.predict(session=sess, x=images)
        print('predictions')
        print(predictions)
        print('')
        test_batches += 1
    test_loss /= test_batches

    print('Test loss: %g' % (test_loss))
