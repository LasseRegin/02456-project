
import os
import data
import math
import utils
import network

import numpy as np
import tensorflow as tf

SHOW_PLOT = 'SHOW_PLOT' in os.environ

# Training parameters
NUM_EPOCHS      = int(os.environ.get('NUM_EPOCHS', 20))
LEARNING_RATE   = float(os.environ.get('LEARNING_RATE', 1e-3))

MAX_VIDEOS = math.inf
if 'RUNNING_ON_LOCAL' in os.environ:
    MAX_VIDEOS = 4

# Intialize frame loader
frame_loader = data.FrameLoader(max_videos=MAX_VIDEOS)
height, width = frame_loader.data.target_height, frame_loader.data.target_width
cells_x = frame_loader.cells_x
cells_y = frame_loader.cells_y

# Split in train and validation
frame_loader = utils.ValidationMinibatches(frame_iterator=frame_loader, cache=frame_loader.data_can_fit_in_memory())

# Setup network
nn = network.ConvolutionalClassifier(name='conv-model-1',
                                     input_shape=(None, height, width),
                                     target_shape=(None, cells_x * cells_y + 1), learning_rate=LEARNING_RATE,
                                     verbose=True)

with tf.Session() as sess:
    nn.init(sess)

    lossTracker = utils.LossTracker(name=nn.name, num_epochs=NUM_EPOCHS, verbose=True)
    for epoch in range(0, NUM_EPOCHS):

        train_loss = 0.
        train_batches = 0
        for images, targets in frame_loader.train:
            train_loss += nn.train_op(session=sess, x=images, y=targets)
            train_batches += 1
        train_loss /= train_batches

        val_loss = 0.
        val_batches = 0
        for images, targets in frame_loader.val:
            val_loss += nn.val_op(session=sess, x=images, y=targets)
            val_batches += 1
        val_loss /= val_batches

        lossTracker.addEpoch(train_loss=train_loss, val_loss=val_loss)

    # Save model
    nn.save(sess)

    # Finally evaluate on test data
    test_loss = 0.
    test_batches = 0
    for images, targets in frame_loader.test:
        test_loss += nn.val_op(session=sess, x=images, y=targets)
        test_batches += 1
    test_loss /= test_batches

    lossTracker.addFinalTestLoss(test_loss)
    lossTracker.save()
