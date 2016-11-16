
import os
import sys
import math
import numpy as np
import tensorflow as tf
import collections

import data
import utils
import network

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

    # Load saved model
    nn.load(sess)

    # The test data will be the same as during training because we fixed the
    # random seed.
    counts = 0
    FP = FN = TP = TN = 0
    last_idx = cells_x * cells_y
    top_k_counter = collections.Counter()
    for input_seqs, targets in frame_loader.test:
        predictions = nn.predict(session=sess, x=input_seqs)

        for prediction, target in zip(predictions, targets):

            # Get predicted index
            y_idx = target.argmax()
            y_hat_idx = prediction.argmax()

            if y_hat_idx == last_idx:
                # Predicts there is no ball
                if y_idx == last_idx:
                    # True negative
                    TN += 1
                else:
                    # False negative
                    FN += 1
            else:
                # Predicts there is a ball
                if y_idx == last_idx:
                    # False positive
                    FP += 1
                else:
                    TP += 1
                    # Get 2d coordinates
                    y_hat_row = y_hat_idx // cells_x
                    y_hat_col = y_hat_idx - y_hat_row
                    y_row = y_idx // cells_x
                    y_col = y_idx - y_row

                    # Compute 1-norm of vector diff
                    norm = int(np.linalg.norm(np.array([y_hat_row - y_row, y_hat_col - y_col]), ord=1))
                    top_k_counter[norm] += 1
            counts += 1


    print('FP: %.4f' % (FP / counts))
    print('TN: %.4f' % (TN / counts))
    print('FN: %.4f' % (FN / counts))

    total_count = sum(count for count in top_k_counter.values())

    # Compute cumulative count
    print('')
    print('Top k CDF (%.4f)' % (TP / counts))
    for k in range(0, max(top_k_counter.keys())):
        counts = 0
        for _k, count in top_k_counter.items():
            if _k <= k: counts += count
        print('k=%d, %.2f' % (k, counts / total_count))
        if k > 3:   break
