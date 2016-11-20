
import os
import data
import math
import utils
import network

import numpy as np
import tensorflow as tf

SHOW_PLOT = 'SHOW_PLOT' in os.environ

# Training parameters
NUM_EPOCHS      = int(os.environ.get('NUM_EPOCHS', 500))
LEARNING_RATE   = float(os.environ.get('LEARNING_RATE', 1e-5))
K_FOLDS         = int(os.environ.get('K_FOLDS', 5))

MAX_VIDEOS = math.inf
if 'RUNNING_ON_LOCAL' in os.environ:
    MAX_VIDEOS = 4

# Intialize frame loader
frame_loader = data.FeatureLoader(max_videos=MAX_VIDEOS)
input_size = frame_loader.BOTTLENECK_TENSOR_SIZE
cells_x = frame_loader.cells_x
cells_y = frame_loader.cells_y

name = 'simple-features-model-1'
error_tracker = utils.ErrorCalculations(name=name)
for K in range(0, K_FOLDS):

    # Split in train, validation and test
    frame_loader_cv = utils.ValidationMinibatches(
        frame_iterator=frame_loader,
        cache=frame_loader.data_can_fit_in_memory(),
        random_seed=K
    )

    # Setup network
    nn = network.LogisticClassifier(name=name,
                                    input_shape=(None, input_size),
                                    target_shape=(None, cells_x * cells_y + 1), learning_rate=LEARNING_RATE,
                                    verbose=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        nn.init(sess)

        for epoch in range(0, NUM_EPOCHS):

            train_loss = 0.
            train_batches = 0
            for features, targets in frame_loader_cv.train:
                train_loss += nn.train_op(session=sess, x=features / features.std(), y=targets)
                train_batches += 1
            train_loss /= train_batches

            val_loss = 0.
            val_batches = 0
            for features, targets in frame_loader_cv.val:
                val_loss += nn.val_op(session=sess, x=features / features.std(), y=targets)
                val_batches += 1
            val_loss /= val_batches

        # Save model
        nn.save(sess)

        # Finally evaluate on test data
        test_loss = 0.
        test_batches = 0
        for features, targets in frame_loader_cv.test:
            test_loss += nn.val_op(session=sess, x=features / features.std(), y=targets)
            test_batches += 1
        test_loss /= test_batches

        error_tracker.evaluate(sess, nn, frame_loader_cv.test, cells_x, cells_y, test_loss)

# Save errors
error_tracker.save()
