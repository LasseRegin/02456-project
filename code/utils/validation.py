
import collections
import numpy as np

class Validation:
    def __init__(self, frame_iterator, test_fraction=0.33):
        """
            Splits selector into train and test data.
        """
        self.train = TrainIterator(frame_iterator=frame_iterator, test_fraction=test_fraction)
        self.test  = TestIterator(frame_iterator=frame_iterator,  test_fraction=test_fraction)


class Iterator:
    def __init__(self, frame_iterator, test_fraction=0.33):
        self.frame_iterator = frame_iterator
        self.test_fraction = test_fraction


class TrainIterator(Iterator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        counter = SeperationCounter()
        for data in self.frame_iterator:
            if counter.test_ratio() >= self.test_fraction:
                counter.increment_train()
                yield data
            else:
                counter.increment_test()


class TestIterator(Iterator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __iter__(self):
        counter = SeperationCounter()
        for data in self.frame_iterator:
            if counter.test_ratio() < self.test_fraction:
                counter.increment_test()
                yield data
            else:
                counter.increment_train()


class SeperationCounter:
    def __init__(self):
        # Count labels in each subset
        self.in_test = 0
        self.in_train = 0

    def test_ratio(self):
        if self.in_test + self.in_train == 0: return 0
        return self.in_test / (self.in_test + self.in_train)

    def increment_train(self):
        self.in_train += 1

    def increment_test(self):
        self.in_test += 1


#
# HDF5 section
#


class HDF5Validation:
    def __init__(self, frame_iterator, count, test_fraction=0.33):
        indices = np.random.permutation(count)
        n_test = int(count // (1.0 / test_fraction))

        idx_test  = sorted(indices[0:n_test].tolist())
        idx_train = sorted(indices[n_test:].tolist())

        self.train = HDF5Iterator(frame_iterator, idx_train)
        self.test  = HDF5Iterator(frame_iterator, idx_test)


class HDF5Iterator:
    def __init__(self, frame_iterator, indices):
        self.frame_iterator = frame_iterator
        self.indices = indices

    def __iter__(self):
        for idx in self.indices:
            yield self.frame_iterator.inputs[idx], self.frame_iterator.targets[idx]
