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
from data.frame_loader import FrameLoader, Frame

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class FeatureLoader(FrameLoader):

    # Model parameters
    MODEL_INPUT_WIDTH = 299
    MODEL_INPUT_HEIGHT = 299
    MODEL_INPUT_DEPTH = 3
    BOTTLENECK_TENSOR_SIZE = 2048

    # Names of important tensors.
    # These names are found by inspection of the graph.
    INPUT_TENSOR_NAME = 'inputs:0'
    BOTTLENECK_TENSOR_NAME = 'logits/flatten/Reshape:0'

    # URL to the frozen graph
    GRAPH_URL = 'https://s3-eu-west-1.amazonaws.com/sportcaster-nn/models/pretrained-graphs/inception_v3_imagenet.pb'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Define frozen graph location
        self.MODEL_PATH = os.path.join(self.DATA_FOLDER, 'inception_v3_imagenet.pb')
        if not os.path.isfile(self.MODEL_PATH):
            print('Downloading frozen graph')
            urlretrieve(self.GRAPH_URL, self.MODEL_PATH)

        # Create memmory mapped numpy arrays
        self.inputs_memmap_filename = os.path.join(self.DATA_FOLDER,  '%s-feature-inputs.dat'  % (self.data_id))
        self.inputs_memmap_size  = (self.frame_count, self.BOTTLENECK_TENSOR_SIZE)

        if not os.path.isfile(self.inputs_memmap_filename):
            # Create numpy memmap file
            print('Creating features input numpy memmap file..')

            # import libraries
            import tensorflow as tf
            from tensorflow.python.platform import gfile

            inputs_memmap = np.memmap(
                filename=self.inputs_memmap_filename,
                dtype='float32',
                mode='w+',
                shape=self.inputs_memmap_size
            )

            # Load frozen inception graph
            with gfile.FastGFile(self.MODEL_PATH, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

                bottleneck_tensor, input_tensor = tf.import_graph_def(
                    graph_def,
                    name='',
                    return_elements=[self.BOTTLENECK_TENSOR_NAME, self.INPUT_TENSOR_NAME]
                )

            # Compute features and write them into memmap
            with tf.Session() as sess:
                for i, frame in enumerate(self.get_frames()):
                    image = frame.image
                    image_features = sess.run(bottleneck_tensor, feed_dict={
                        input_tensor: image.reshape((1,) + image.shape)
                    })

                    image_features = image_features.flatten().astype('float32')
                    inputs_memmap[i, :] = image_features
            inputs_memmap.flush()
            del inputs_memmap

        self.inputs_memmap = np.memmap(
            filename=self.inputs_memmap_filename,
            dtype='float32',
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
        print('FeatureLoader __iter__ called')

        for i in range(0, self.frame_count):
            yield self.inputs_memmap[i], self.targets_memmap[i]
