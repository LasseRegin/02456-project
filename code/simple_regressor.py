
import os
import data
import utils
import network

import numpy as np
import tensorflow as tf

SHOW_PLOT = 'SHOW_PLOT' in os.environ

# Training parameters
NUM_EPOCHS = 10
LEARNING_RATE = 1e-5

# Intialize frame loader
frame_loader = data.FrameLoader(shuffle=True, validation_group='train')
height, width = frame_loader.data.target_height, frame_loader.data.target_width

# Split in train and validation
frame_loader = utils.ValidationMinibatches(frame_iterator=frame_loader)

# Setup network
nn = network.SimpleRegressor(name='simple-model-1',
                             input_shape=(None, height, width),
                             target_shape=(None, 2), learning_rate=LEARNING_RATE,
                             verbose=True)

with tf.Session() as sess:
    nn.init(sess)

    plot = utils.LossPlot(show=SHOW_PLOT)
    for epoch in range(0, NUM_EPOCHS):

        train_loss = 0.
        train_batches = 0
        for images, targets in frame_loader.train:
            images /= images.std() # TODO: do somewhere else
            train_loss += nn.train_op(session=sess, x=images, y=targets)
            train_batches += 1
        train_loss /= train_batches

        val_loss = 0.
        val_batches = 0
        for images, targets in frame_loader.val:
            images /= images.std() # TODO: do somewhere else
            val_loss += nn.val_op(session=sess, x=images, y=targets)
            val_batches += 1
        val_loss /= val_batches

        plot.update(epoch=epoch, loss=train_loss)

        print('Epoch %d/%d' % (epoch+1, NUM_EPOCHS))
        print('Train loss: %g' % (train_loss))
        print('Val loss: %g' % (val_loss))


    # Save model
    nn.save(sess)
