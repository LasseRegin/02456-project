from __future__ import division, generators, print_function, unicode_literals, with_statement

import os
import sys
import json
import math
import psutil
import random
import imageio
import numpy as np

from data.persistence import DataPersistence

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class FrameLoader:
    DATA_FOLDER = os.path.join(FILEPATH, 'raw')
    MEMMAP_FILE = os.path.join(DATA_FOLDER, 'memmap_file.dat')

    def __init__(self, cells_x=20, cells_y=12, **kwargs):

        # Heatmap dimensions
        self.cells_x = cells_x
        self.cells_y = cells_y

        # Check data persistency
        self.data = DataPersistence(**kwargs)

        # Get unique identifier for specific data
        self.data_id = str(hash(self.data))

        # Load frame filenames
        self.frames = []
        for video in self.data.videos:
            with open(video['annotation'], 'r') as f:
                annotation = json.load(f)

            balls = annotation['balls']
            for i in range(0, video['frame_count']):
                # Get ball info
                ball = balls.get(str(i), None)
                found = ball is not None

                if found:
                    x = ball['x'] / self.data.ORIGINAL_WIDTH
                    y = ball['y'] / self.data.ORIGINAL_HEIGHT
                else:
                    x = None
                    y = None


                # Define frame filename
                frame_filename = '%s/%s.png' % (video['foldername'], i + 1)

                self.frames.append(Frame(
                    x=x,
                    y=y,
                    found=found,
                    filename=frame_filename,
                    foldername=video['foldername']
                ))

        # Frame count
        self.frame_count = len(self.frames)

        # Create memmory mapped numpy arrays
        self.inputs_memmap_filename = os.path.join(self.DATA_FOLDER, '%s-inputs.dat' % (self.data_id))
        self.targets_memmap_filename = os.path.join(self.DATA_FOLDER, '%s-targets-%d-%d.dat' % (self.data_id, self.cells_x, self.cells_y))
        self.inputs_memmap_size = (self.frame_count, self.data.target_height, self.data.target_width, 3)
        self.targets_memmap_size = (self.frame_count, self.cells_x * self.cells_y + 1)

        if not os.path.isfile(self.inputs_memmap_filename):
            # Create numpy memmap file
            print('Creating inputs numpy memmap file..')
            inputs_memmap = np.memmap(
                filename=self.inputs_memmap_filename,
                dtype='uint8',
                mode='w+',
                shape=self.inputs_memmap_size
            )

            # Write frames into memmap
            for i, frame in enumerate(self.get_frames()):
                #image_preprocessed = frame.image - frame.image.mean()
                #inputs_memmap[i, :, :, :] = image_preprocessed.astype('float32')
                inputs_memmap[i, :, :, :] = frame.image
            inputs_memmap.flush()
            del inputs_memmap

        if not os.path.isfile(self.targets_memmap_filename):
            # Create numpy memmap file
            print('Creating targets numpy memmap file..')
            targets_memmap = np.memmap(
                filename=self.targets_memmap_filename,
                dtype='float32',
                mode='w+',
                shape=self.targets_memmap_size
            )

            for i, frame in enumerate(self.get_frames()):
                targets_memmap[i, :] = ballPositionHeatMap(
                    found=frame.found,
                    x=frame.x,
                    y=frame.y,
                    cells_x=self.cells_x,
                    cells_y=self.cells_y
                )

            targets_memmap.flush()
            del targets_memmap

        self.inputs_memmap = np.memmap(
            filename=self.inputs_memmap_filename,
            dtype='uint8',
            mode='c',
            shape=self.inputs_memmap_size
        )

        self.targets_memmap = np.memmap(
            filename=self.targets_memmap_filename,
            dtype='float32',
            mode='c',
            shape=self.targets_memmap_size
        )


    def __iter__(self):
        print('FrameLoader __iter__ called')

        for i in range(0, self.frame_count):
            yield self.inputs_memmap[i], self.targets_memmap[i]

    def get_frames(self):
        for frame in self.frames:
            # Read file
            frame.image = imageio.imread(frame.filename)
            yield frame


    def available_memory(self):
        """
            Returns the amount of available memory in bytes.
        """
        #mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        mem_bytes = psutil.virtual_memory().available
        return mem_bytes


    def data_can_fit_in_memory(self):
        # Determine size of a single element in bytes
        element_size = self.inputs_memmap.dtype.itemsize

        # Get number of elements in total
        element_count = self.inputs_memmap.size

        # Total size in bytes
        size_total = element_count * element_size

        # Get available memory
        memory_available = self.available_memory()

        # Determine if we have enough memory (with a buffer of 1 GB)
        memory_diff = memory_available - size_total
        memory_diff_gb = memory_diff / (1024 ** 3)

        return memory_diff_gb > 1.0



class Frame:
    def __init__(self, x, y, found, filename, foldername, image=None):
        self.x = x
        self.y = y
        self.found = found
        self.filename = filename
        self.foldername = foldername # Used for identifying what video the frame is from
        self.image = image


ballPositionHeatMapWeights = np.array([
    [0.18, 0.25, 0.18],
    [0.25, 1.00, 0.25],
    [0.18, 0.25, 0.18]
])


def ballPositionHeatMap(found, x, y, cells_x, cells_y):
    heatmap = np.zeros(shape=(cells_y, cells_x))
    if not found:
        return np.hstack((heatmap.flatten(), 1.0)).astype('float32')

    # Get ball cell coordinate
    x_cell = math.floor(x * cells_x)
    y_cell = math.floor(y * cells_y)

    for w_x, x_offset in enumerate([-1, 0, 1]):
        for w_y, y_offset in enumerate([-1, 0, 1]):
            x_idx = x_cell + x_offset
            y_idx = y_cell + y_offset

            # Check border constraints
            if x_idx < 0 or y_idx < 0:  continue
            if x_idx + 1 > cells_x:     continue
            if y_idx + 1 > cells_y:     continue

            # Assign weight
            heatmap[y_idx, x_idx] = ballPositionHeatMapWeights[w_y, w_x]

    return np.hstack((heatmap.flatten(), 0.0)).astype('float32')
