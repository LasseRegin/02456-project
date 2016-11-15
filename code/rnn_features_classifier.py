
import os
import data
import math
import utils
import network

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

SHOW_PLOT = 'SHOW_PLOT' in os.environ

# Training parameters
NUM_EPOCHS      = int(os.environ.get('NUM_EPOCHS', 10))
LEARNING_RATE   = float(os.environ.get('LEARNING_RATE', 1e-6))
N_STEPS         = int(os.environ.get('N_STEPS', 10))
BATCH_SIZE      = int(os.environ.get('BATCH_SIZE', 50))
KEEP_PROB       = float(os.environ.get('KEEP_PROB', 1.0))

MAX_VIDEOS = math.inf
if 'RUNNING_ON_LOCAL' in os.environ:
    MAX_VIDEOS = 4

# Intialize frame loader
frame_loader = data.FeatureSequenceLoader(max_videos=MAX_VIDEOS, n_steps=N_STEPS)
input_size = frame_loader.BOTTLENECK_TENSOR_SIZE
cells_x = frame_loader.cells_x
cells_y = frame_loader.cells_y


# Split in train, validation and test set
frame_loader = utils.SequenceValidationMinibatches(frame_iterator=frame_loader, batch_size=BATCH_SIZE)

# Initialize network
nn = network.RNNClassifier(name='rnn-features-model-1',
                           n_steps=N_STEPS,
                           keep_prob=KEEP_PROB,
                           input_shape=(None, N_STEPS, input_size),
                           target_shape=(None, cells_x * cells_y + 1),
                           learning_rate=LEARNING_RATE,
                           verbose=True)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    nn.init(sess)

    lossTracker = utils.LossTracker(name=nn.name, num_epochs=NUM_EPOCHS, verbose=True)
    for epoch in range(0, NUM_EPOCHS):

        train_loss = 0.
        train_batches = 0
        for input_seqs, targets in frame_loader.train:
            train_loss += nn.train_op(session=sess, x=input_seqs, y=targets)
            train_batches += 1
        train_loss /= train_batches

        val_loss = 0.
        val_batches = 0
        for input_seqs, targets in frame_loader.val:
            val_loss += nn.val_op(session=sess, x=input_seqs, y=targets)
            val_batches += 1
        val_loss /= val_batches

        lossTracker.addEpoch(train_loss=train_loss, val_loss=val_loss)

    # Save model
    nn.save(sess)

    # Finally evaluate on test data
    test_loss = 0.
    test_batches = 0
    for input_seqs, targets in frame_loader.test:
        test_loss += nn.val_op(session=sess, x=input_seqs, y=targets)
        test_batches += 1
    test_loss /= test_batches

    lossTracker.addFinalTestLoss(test_loss)
    lossTracker.save()
