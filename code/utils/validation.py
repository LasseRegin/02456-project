
import math
import random
import collections
import numpy as np


class ValidationMinibatches:
    def __init__(self, frame_iterator, val_fraction=0.20, test_fraction=0.30, batch_size=50, cache=False):
        """
            Splits frame loader into train and validation data.
        """
        self.frame_iterator = frame_iterator
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.cache = cache

        if not hasattr(frame_iterator, 'inputs_memmap'):
            raise KeyError('frame_iterator must be of type FrameLoader')

        frame_count = self.frame_iterator.frame_count

        self.n_val = int(frame_count * self.val_fraction)
        self.n_test = int(frame_count * self.test_fraction)
        self.n_train = frame_count - self.n_val - self.n_test
        order = np.random.permutation(frame_count)

        self.order_train = order[0:self.n_train]
        self.order_val   = order[self.n_train:self.n_train+self.n_val]
        self.order_test  = order[self.n_train+self.n_val:]

        self.batch_count_train = math.ceil(self.n_train / self.batch_size)
        self.batch_count_val   = math.ceil(self.n_val   / self.batch_size)
        self.batch_count_test  = math.ceil(self.n_test  / self.batch_size)


    @property
    def train(self):
        # Shuffle indices
        random.shuffle(self.order_train)

        for batch_number in range(0, self.batch_count_train):
            idx_from = batch_number * self.batch_size
            idx_to   = min((batch_number + 1) * self.batch_size, self.n_train)
            indices  = self.order_train[idx_from:idx_to]

            yield self.frame_iterator.inputs_memmap[indices], self.frame_iterator.targets_memmap[indices]


    @property
    def val(self):
        # Shuffle indices
        random.shuffle(self.order_val)

        for batch_number in range(0, self.batch_count_val):
            idx_from = batch_number * self.batch_size
            idx_to   = min((batch_number + 1) * self.batch_size, self.n_val)
            indices  = self.order_val[idx_from:idx_to]

            yield self.frame_iterator.inputs_memmap[indices], self.frame_iterator.targets_memmap[indices]


    @property
    def test(self):
        # Shuffle indices
        random.shuffle(self.order_test)

        for batch_number in range(0, self.batch_count_test):
            idx_from = batch_number * self.batch_size
            idx_to   = min((batch_number + 1) * self.batch_size, self.n_test)
            indices  = self.order_test[idx_from:idx_to]

            yield self.frame_iterator.inputs_memmap[indices], self.frame_iterator.targets_memmap[indices]
