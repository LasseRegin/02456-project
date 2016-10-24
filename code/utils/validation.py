
import math
import random
import collections
import numpy as np


class ValidationMinibatches:
    def __init__(self, frame_iterator, val_fraction=0.25, batch_size=50, cache=False):
        """
            Splits frame loader into train and validation data.
        """
        self.frame_iterator = frame_iterator
        self.val_fraction = val_fraction
        self.batch_size = batch_size
        self.cache = cache

        if not hasattr(frame_iterator, 'order'):
            raise KeyError('frame_iterator must be of type FrameLoader')

        frame_count = len(self.frame_iterator.order)

        self.n_val = int(frame_count * self.val_fraction)
        self.n_train = frame_count - self.n_val
        order = np.random.permutation(frame_count)

        self.order_train = order[0:self.n_train]
        self.order_val   = order[self.n_train:]

        self.batch_count_train = math.ceil(self.n_train / self.batch_size)
        self.batch_count_val   = math.ceil(self.n_val   / self.batch_size)

        if self.cache:
            print('Loading data into memory..')
            self.inputs  = self.frame_iterator.inputs[...]
            self.targets = self.frame_iterator.targets[...]


    @property
    def train(self):
        # Shuffle indices
        random.shuffle(self.order_train)

        for batch_number in range(0, self.batch_count_train):
            idx_from = batch_number * self.batch_size
            idx_to   = min((batch_number + 1) * self.batch_size, self.n_train)
            indices  = self.order_train[idx_from:idx_to]

            if self.cache:
                yield self.inputs[indices], self.targets[indices]
            else:
                # Sort for faster h5py slicing
                indices = sorted(indices)
                yield self.frame_iterator.inputs[indices], self.frame_iterator.targets[indices]


    @property
    def val(self):
        # Shuffle indices
        random.shuffle(self.order_val)

        for batch_number in range(0, self.batch_count_val):
            idx_from = batch_number * self.batch_size
            idx_to   = min((batch_number + 1) * self.batch_size, self.n_val)
            indices  = self.order_val[idx_from:idx_to]

            if self.cache:
                yield self.inputs[indices], self.targets[indices]
            else:
                # Sort for faster h5py slicing
                indices = sorted(indices)
                yield self.frame_iterator.inputs[indices], self.frame_iterator.targets[indices]



class Validation:
    def __init__(self, frame_iterator, frame_count, test_fraction=0.33):
        """
            Splits selector into train and test data.
        """
        self.frame_iterator = frame_iterator
        self.test_fraction = test_fraction

        n_test = int(frame_count // (1.0 / test_fraction))
        order = np.random.permutation(frame_count)

        self.order_train = order[n_test:]
        self.order_test  = order[0:n_test]


    @property
    def train(self):
        for idx in self.order_train:
            yield self.frame_iterator.inputs[idx], self.frame_iterator.targets[idx]

    @property
    def test(self):
        for idx in self.order_test:
            yield self.frame_iterator.inputs[idx], self.frame_iterator.targets[idx]

#
# class Iterator:
#     def __init__(self, frame_iterator, test_fraction=0.33):
#         self.frame_iterator = frame_iterator
#         self.test_fraction = test_fraction
#
#
# class TrainIterator(Iterator):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def __iter__(self):
#         counter = SeperationCounter()
#         for data in self.frame_iterator:
#             if counter.test_ratio() >= self.test_fraction:
#                 counter.increment_train()
#                 yield data
#             else:
#                 counter.increment_test()
#
#
# class TestIterator(Iterator):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def __iter__(self):
#         counter = SeperationCounter()
#         for data in self.frame_iterator:
#             if counter.test_ratio() < self.test_fraction:
#                 counter.increment_test()
#                 yield data
#             else:
#                 counter.increment_train()
#
#
# class SeperationCounter:
#     def __init__(self):
#         # Count labels in each subset
#         self.in_test = 0
#         self.in_train = 0
#
#     def test_ratio(self):
#         if self.in_test + self.in_train == 0: return 0
#         return self.in_test / (self.in_test + self.in_train)
#
#     def increment_train(self):
#         self.in_train += 1
#
#     def increment_test(self):
#         self.in_test += 1
