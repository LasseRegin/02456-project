
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
K_FOLDS         = int(os.environ.get('K_FOLDS', 5))

MAX_VIDEOS = math.inf
if 'RUNNING_ON_LOCAL' in os.environ:
    MAX_VIDEOS = 4

# Intialize frame loader
frame_loader = data.FeatureSequenceLoader(max_videos=MAX_VIDEOS, n_steps=N_STEPS)
input_size = frame_loader.BOTTLENECK_TENSOR_SIZE
cells_x = frame_loader.cells_x
cells_y = frame_loader.cells_y

name = 'rnn-features-model-1'
error_tracker = utils.ErrorCalculations(name=name)
for K in range(0, K_FOLDS):

    # Split in train, validation and test set
    frame_loader_cv = utils.SequenceValidationMinibatches(
        frame_iterator=frame_loader,
        batch_size=BATCH_SIZE,
        random_seed=K
    )

    # Initialize network
    nn = network.RNNClassifier(name=name,
                               n_steps=N_STEPS,
                               keep_prob=KEEP_PROB,
                               input_shape=(None, N_STEPS, input_size),
                               target_shape=(None, cells_x * cells_y + 1),
                               learning_rate=LEARNING_RATE,
                               verbose=False)



    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        nn.init(sess)

        for epoch in range(0, NUM_EPOCHS):

            train_loss = 0.
            train_batches = 0
            for input_seqs, targets in frame_loader_cv.train:
                train_loss += nn.train_op(session=sess, x=input_seqs, y=targets)
                train_batches += 1
            train_loss /= train_batches

            val_loss = 0.
            val_batches = 0
            for input_seqs, targets in frame_loader_cv.val:
                val_loss += nn.val_op(session=sess, x=input_seqs, y=targets)
                val_batches += 1
            val_loss /= val_batches

        # Save model
        nn.save(sess)

        # Finally evaluate on test data
        test_loss = 0.
        test_batches = 0
        for input_seqs, targets in frame_loader_cv.test:
            test_loss += nn.val_op(session=sess, x=input_seqs, y=targets)
            test_batches += 1
        test_loss /= test_batches

        error_tracker.evaluate(sess, nn, frame_loader_cv.test, cells_x, cells_y, test_loss)

    # Remove nodes from graph
    tf.reset_default_graph()

# Save errors
error_tracker.save()
