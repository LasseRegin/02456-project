
import numpy as np

# TODO: Make the minibatch shuffled

class Minibatch:
    def __init__(self, frame_iterator, batch_size=20):
        self.frame_iterator = frame_iterator
        self.batch_size = batch_size

    def __iter__(self):
        self.counter = 0
        #for image, target in getattr(self.frame_iterator, 'train'):
        for image, target in self.frame_iterator:
            if self.counter == 0:
                image_batch, target_batch = [], []

            image_batch.append(image)
            target_batch.append(target)

            self.counter += 1

            if self.counter == self.batch_size:
                self.counter = 0
                order = np.random.permutation(self.batch_size)
                yield np.array(image_batch)[order], np.array(target_batch)[order]

        # If number of observations doesn't divide `batch_size` return remainding
        if self.counter > 0:
            self.counter = 0
            order = np.random.permutation(len(image_batch))
            yield np.array(image_batch)[order], np.array(target_batch)[order]
