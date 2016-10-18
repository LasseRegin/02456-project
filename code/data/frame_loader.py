from __future__ import division, generators, print_function, unicode_literals, with_statement

import os
import json
import h5py
import imageio
import random
import numpy as np

from data.persistence import DataPersistence

FILEPATH = os.path.dirname(os.path.abspath(__file__))


class FrameLoader:
    DATA_FOLDER = os.path.join(FILEPATH, 'raw')
    HDF5_FILE = os.path.join(DATA_FOLDER, 'data.hdf5')

    def __init__(self, shuffle=False, **kwargs):
        # Check data persistency
        self.shuffle = shuffle
        self.data = DataPersistence(**kwargs)

        # Get unique identifier for specific data
        self.data_id = str(hash(self.data))

        # Check if data is available
        if not self.dataset_available():
            self.create_dataset()

        # Load datafile
        f = h5py.File(self.HDF5_FILE, 'r')
        group = f[self.data_id]
        self.inputs  = group['inputs']
        self.targets = group['targets']

    def __iter__(self):
        #for image, target in zip(self.inputs, self.targets):
        #    yield image, target

        if self.shuffle:
            order = np.random.permutation(self.inputs.shape[0])
        else:
            order = range(0, self.inputs.shape[0])

        for i in order:
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

        try:
            # We need to loop through the frames to determine the size of the dataset
            frame_count = sum(1 for _ in self.get_frames())

            # Initialize datasets
            inputs_data_size  = (frame_count, self.data.target_height, self.data.target_width)
            targets_data_size = (frame_count, 2)
            inputs  = group.create_dataset("inputs",  inputs_data_size,  dtype='float32')
            targets = group.create_dataset("targets", targets_data_size, dtype='float32')

            for i, frame in enumerate(self.get_frames()):
                # Preprocess frame
                image = frame.image.astype('float32')
                image -= image.mean()
                target = np.array([frame.x, frame.y])

                # Save observation
                inputs[i, :, :] = image
                targets[i, :] = target
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


class Frame:
    def __init__(self, x, y, image):
        self.x = x
        self.y = y
        self.image = image
