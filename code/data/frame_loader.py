from __future__ import division, generators, print_function, unicode_literals, with_statement

import os
import sys
import json
import h5py
import psutil
import imageio
import random
import numpy as np

from data.persistence import DataPersistence

FILEPATH = os.path.dirname(os.path.abspath(__file__))


class FrameLoader:
    DATA_FOLDER = os.path.join(FILEPATH, 'raw')
    HDF5_FILE = os.path.join(DATA_FOLDER, 'data.hdf5')

    def __init__(self, shuffle=True, validation_group='train', **kwargs):
        # Check data persistency
        self.data = DataPersistence(**kwargs)

        # Get unique identifier for specific data
        self.data_id = str(hash(self.data))

        # Check if data is available
        if not self.dataset_available():
            self.create_dataset()

        # Load datafile
        f = h5py.File(self.HDF5_FILE, 'r')
        group = f[self.data_id]

        if not validation_group in ['train', 'test']:
            raise KeyError('Wrong validation_group key provided')

        val_group = group[validation_group]
        self.inputs  = val_group['inputs']
        self.targets = val_group['targets']

        # Determine number of frames in dataset
        self.frame_count = self.inputs.shape[0]
        self.dtype = self.inputs.dtype
        self.shape = self.inputs.shape

        # Determine order of frames
        if shuffle:
            self.order = np.random.permutation(self.frame_count)
        else:
            self.order = range(0, self.frame_count)

    def __iter__(self):
        print('FrameLoader __iter__ called')

        for i in self.order:
            yield self.inputs[i], self.targets[i]

    def dataset_available(self):
        f = h5py.File(self.HDF5_FILE, 'a')
        is_available = self.data_id in f
        f.close()
        return is_available

    def create_dataset(self):
        print('Creating dataset..')

        # Load file with read/write permissions
        f = h5py.File(self.HDF5_FILE, 'a')

        # Create group for the dataset
        group = f.create_group(self.data_id)

        # Create Training and Test group
        group_train = group.create_group('train')
        group_test  = group.create_group('test')

        try:
            # We need to loop through the frames to determine the size of the dataset
            frame_count = sum(1 for _ in self.get_frames())

            # Take away 1/4 of the data for testing
            n_test  = int(frame_count // 4)
            n_train = frame_count - n_test

            # Initialize datasets
            inputs_data_size_train  = (n_train, self.data.target_height, self.data.target_width)
            targets_data_size_train = (n_train, 2)
            inputs_train  = group_train.create_dataset('inputs',  inputs_data_size_train,  dtype='float32')
            targets_train = group_train.create_dataset('targets', targets_data_size_train, dtype='float32')

            inputs_data_size_test  = (n_test, self.data.target_height, self.data.target_width)
            targets_data_size_test = (n_test, 2)
            inputs_test  = group_test.create_dataset('inputs',  inputs_data_size_test,  dtype='float32')
            targets_test = group_test.create_dataset('targets', targets_data_size_test, dtype='float32')

            for i, frame in enumerate(self.get_frames()):
                # Preprocess frame
                image = frame.image.astype('float32')
                image -= image.mean()
                target = np.array([frame.x, frame.y])

                # Save observation
                if i >= n_train:
                    idx = i % n_train
                    inputs_test[idx, :, :] = image
                    targets_test[idx, :] = target
                else:
                    inputs_train[i, :, :] = image
                    targets_train[i, :] = target

        except Exception:
            # If anything went wrong delete the group again
            del f[self.data_id]
        except KeyboardInterrupt:
            del f[self.data_id]

        # Close file
        f.close()


    def get_frames(self):
        for video in self.data.videos:
            with open(video['annotation']) as f:
                data = json.load(f)
            balls = data['balls']

            # Initialize video capture
            image_reader = imageio.get_reader(uri=video['filename'], format='ffmpeg')

            for i, image in enumerate(image_reader):

                # Get frame information
                ball = balls.get(str(i), None)
                found = ball is not None

                # Continue if we ball was not found
                if not found:   continue

                # Convert to grayscale
                image = image.mean(axis=2)

                yield Frame(
                    image=image,
                    x=ball['x'] / self.data.ORIGINAL_WIDTH,
                    y=ball['y'] / self.data.ORIGINAL_HEIGHT
                )


    def available_memory(self):
        """
            Returns the amount of available memory in bytes.
        """
        #mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        mem_bytes = psutil.virtual_memory().available
        return mem_bytes


    def data_can_fit_in_cache(self):
        # Determine size of a single element in bytes
        element_size = self.inputs.dtype.itemsize

        # Get number of elements in total
        element_count = self.inputs.size

        # Total size in bytes
        size_total = element_count * element_size

        # Get available memory
        memory_available = self.available_memory()

        # Determine if we have enough memory (with a buffer of 1 GB)
        memory_diff = memory_available - size_total
        memory_diff_gb = memory_diff / (1024 ** 3)

        print('size_total')
        print(size_total)
        print('memory_available')
        print(memory_available)
        print('memory_diff')
        print(memory_diff)
        print('memory_diff_gb')
        print(memory_diff_gb)

        return memory_diff_gb > 1.0



class Frame:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
