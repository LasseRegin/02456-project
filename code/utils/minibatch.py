
import numpy as np
import random

class MinibatchUncached:
    """
        Used when we cannot fit the entire dataset in memory.
    """
    def __init__(self, frame_iterator, batch_size=50, input_type='float32',
                 target_type='float32'):
        self.frame_iterator = frame_iterator
        self.batch_size = batch_size
        self.input_type = input_type
        self.target_type = target_type

    def quick_shuffle(self, image_batch, target_batch):
        seed = random.random()
        random.shuffle(image_batch,  lambda: seed)
        random.shuffle(target_batch, lambda: seed)

    def __iter__(self):
        self.counter = 0
        #for image, target in getattr(self.frame_iterator, 'train'):
        for image, target in self.frame_iterator:
            if self.counter == 0:
                image_batch, target_batch = [], []

            image_batch.append(image)
            target_batch.append(target)
            #print(target_batch)

            self.counter += 1

            if self.counter == self.batch_size:
                self.counter = 0
                #yield image_batch, target_batch
                #order = np.random.permutation(self.batch_size)
                #yield np.array(image_batch)[order], np.array(target_batch)[order]
                #self.quick_shuffle(image_batch, target_batch)
                #seed = random.random()
                #random.shuffle(image_batch,  lambda: seed)
                #random.shuffle(target_batch, lambda: seed)
                #print('image_batch')
                #print(image_batch)
                yield np.asarray(image_batch, dtype=self.input_type), np.asarray(target_batch, dtype=self.target_type)

        # If number of observations doesn't divide `batch_size` return remainding
        if self.counter > 0:
            self.counter = 0
            #order = np.random.permutation(len(image_batch))
            #self.quick_shuffle(image_batch, target_batch)
            #yield np.array(image_batch)[order], np.array(target_batch)[order]
            yield np.asarray(image_batch, dtype=self.input_type), np.asarray(target_batch, dtype=self.target_type)
            #yield np.asarray(image_batch), np.asarray(target_batch)


# NOTE: Taken from
# https://github.com/AndreasMadsen/course-02460/blob/master/code/helpers/minibatch.py
class Minibatch:
    def __init__(self, data_iterable, batchsize=50,
                 input_type='float32', target_type='float32'):
        self._data_iterable = data_iterable
        self._batchsize = batchsize
        self._input_type = input_type
        self._target_type = target_type

        input_data = []
        target_data = []
        for input, target in data_iterable:
            input_data.append(input)
            target_data.append(target)

        self._input_cache  = np.asarray(input_data,  dtype=self._input_type)
        self._target_cache = np.asarray(target_data, dtype=self._target_type)

    def __iter__(self):
        return MinibatchCache(self._input_cache, self._target_cache,
                              self._batchsize)

    @property
    def data(self):
        return (self._input_cache, self._target_cache)

class MinibatchCache:
    def __init__(self, input_cache, target_cache, batchsize):
        self._input_cache = input_cache
        self._target_cache = target_cache

        self._size = input_cache.shape[0]
        self._batchsize = batchsize

        self._order = np.random.permutation(self._size)
        self._position = 0

    def __next__(self):
        if (self._position >= self._size): raise StopIteration

        curr_position = self._position
        next_position = self._position + self._batchsize

        # Select batch by only copying the data subset
        input_batch = self._input_cache[self._order[curr_position:next_position]]
        target_batch = self._target_cache[self._order[curr_position:next_position]]

        # Move position for next iteration
        self._position = next_position

        return (input_batch, target_batch)
