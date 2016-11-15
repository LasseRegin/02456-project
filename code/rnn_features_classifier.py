
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



for input_seqs, targets in frame_loader.train:
    print(input_seqs.shape)
    print(targets.shape)

for input_seqs, targets in frame_loader.val:
    print(input_seqs.shape)
    print(targets.shape)

for input_seqs, targets in frame_loader.test:
    print(input_seqs.shape)
    print(targets.shape)


import sys; sys.exit()

#
#
# class Minibatches:
#     def __init__(self, frame_seq_loader, batch_size=20):
#         self.input_batches = []
#         self.target_batches = []
#         self.batch_size = batch_size
#         inputs_batch = []
#         target_batch = []
#         count = 0
#         for i, (frame_seq, target) in enumerate(frame_seq_loader):
#             inputs_batch.append(frame_seq)
#             target_batch.append(target)
#             count += 1
#             if count == self.batch_size:
#                 self.input_batches.append(np.asarray(inputs_batch))
#                 self.target_batches.append(np.asarray(target_batch))
#                 inputs_batch = []
#                 target_batch = []
#                 count = 0
#         if count > 0:
#             self.input_batches.append(np.asarray(inputs_batch))
#             self.target_batches.append(np.asarray(target_batch))
#             inputs_batch = []
#             target_batch = []
#
#     def __iter__(self):
#         for inputs, targets in zip(self.input_batches, self.target_batches):
#             yield inputs, targets
#
# frame_loader = Minibatches(frame_seq_loader=frame_loader, batch_size=BATCH_SIZE)

# TODO: Make validation work
## Split in train, validation and test
#frame_loader = utils.ValidationMinibatches(frame_iterator=frame_loader, cache=frame_loader.data_can_fit_in_memory())


# Initialize network
nn = network.RNNClassifier(name='rnn-features-model-1',
                           n_steps=N_STEPS,
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
            print(input_seqs)
            print(targets)
            train_loss += nn.train_op(session=sess, x=input_seqs, y=targets)
            train_batches += 1
        train_loss /= train_batches

        val_loss = 0.
        val_batches = 0
        for input_seqs, targets in frame_loader.val:
            print(input_seqs)
            print(targets)
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
