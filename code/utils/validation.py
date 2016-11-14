
import math
import random
import collections
import numpy as np


class ValidationMinibatches:
    def __init__(self, frame_iterator, val_fraction=0.20, test_fraction=0.30,
                 batch_size=50, cache=False, random_seed=42):
        """
            Splits frame loader into train and validation data.
        """
        self.frame_iterator = frame_iterator
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.cache = cache

        # Set random seed so we get consistent results
        np.random.seed(random_seed)

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



class SequenceValidationMinibatches:
    def __init__(self, frame_iterator, val_fraction=0.20, test_fraction=0.30,
                 batch_size=50, random_seed=42):
        """
            Splits frame sequence loader into train and validation data.
        """
        self.frame_iterator = frame_iterator
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.batch_size = batch_size
        self.n_steps = self.frame_iterator.n_steps

        # Set random seed so we get consistent results
        np.random.seed(random_seed)

        if not hasattr(frame_iterator, 'frame_seq_indices'):
            raise KeyError('frame_iterator must be of type FeatureSequenceLoader')

        # Get number of set of indicies
        indices_count = len(self.frame_iterator.frame_seq_indices)

        # We need to leave out a gap between train, validation and test
        # so we don't overlap the indicies (and thereby introduce data-snooping).
        self.n_val   = int(indices_count * self.val_fraction)
        self.n_test  = int(indices_count * self.test_fraction)
        self.n_train = indices_count - self.n_val - self.n_test

        order = np.arange(0, indices_count)

        train_from       = 0
        train_to         = self.n_train - self.n_steps
        self.order_train = order[train_from:train_to]

        val_from         = train_to + self.n_steps
        val_to           = val_from + self.n_val - self.n_steps
        self.order_val   = order[val_from:val_to]

        test_from       = val_to + self.n_steps
        test_to         = test_from + self.n_test
        self.order_test = order[test_from:test_to]

        self.batch_count_train = math.ceil((self.n_train               ) / self.batch_size)
        self.batch_count_val   = math.ceil((self.n_val   - self.n_steps) / self.batch_size)
        self.batch_count_test  = math.ceil((self.n_test  - self.n_steps) / self.batch_size)



    @property
    def train(self):
        # Shuffle indices
        random.shuffle(self.order_train)

        for batch_number in range(0, self.batch_count_train):
            idx_from = batch_number * self.batch_size
            idx_to   = min((batch_number + 1) * self.batch_size, self.n_train - self.n_steps)
            indices  = self.order_train[idx_from:idx_to]

            indices = self.frame_iterator.frame_seq_indices[indices]
            indices_target = indices[:, -1]

            yield self.frame_iterator.inputs_memmap[indices], self.frame_iterator.targets_memmap[indices_target]


    @property
    def val(self):
        # Shuffle indices
        random.shuffle(self.order_val)

        for batch_number in range(0, self.batch_count_val):
            idx_from = batch_number * self.batch_size
            idx_to   = min((batch_number + 1) * self.batch_size, self.n_val - self.n_steps)
            indices  = self.order_val[idx_from:idx_to]

            indices = self.frame_iterator.frame_seq_indices[indices]
            indices_target = indices[:, -1]

            yield self.frame_iterator.inputs_memmap[indices], self.frame_iterator.targets_memmap[indices_target]


    @property
    def test(self):
        # Shuffle indices
        random.shuffle(self.order_test)

        for batch_number in range(0, self.batch_count_test):
            idx_from = batch_number * self.batch_size
            idx_to   = min((batch_number + 1) * self.batch_size, self.n_test)
            indices  = self.order_test[idx_from:idx_to]

            indices = self.frame_iterator.frame_seq_indices[indices]
            indices_target = indices[:, -1]

            yield self.frame_iterator.inputs_memmap[indices], self.frame_iterator.targets_memmap[indices_target]
