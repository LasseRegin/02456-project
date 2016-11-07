from __future__ import print_function, division, absolute_import

import os
import math
import tensorflow as tf
from tensorflow.python.platform import gfile

import tensorflow.contrib.slim as slim

import data

# Input, bottleneck and output size.
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_OUTPUT_SIZE = 2

# Names of important tensors.
# These names are found by inspection of the graph.
INPUT_TENSOR_NAME = 'inputs:0'
BOTTLENECK_TENSOR_NAME = 'logits/flatten/Reshape:0'
FINAL_TENSOR_NAME = 'final_results'

# Hyperparameters.
LEARNING_RATE = 0.01


FILEPATH = os.path.dirname(os.path.abspath(__file__))
MODELS_PATH = os.path.join(FILEPATH, 'network/models')
MODEL_PATH = os.path.join(MODELS_PATH, 'inception_v3_imagenet.pb')

# Load frozen inception graph
with gfile.FastGFile(MODEL_PATH, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

    bottleneck_tensor, input_tensor = tf.import_graph_def(
        graph_def,
        name='',
        return_elements=[BOTTLENECK_TENSOR_NAME, INPUT_TENSOR_NAME]
    )


MAX_VIDEOS = math.inf
if 'RUNNING_ON_LOCAL' in os.environ:
    MAX_VIDEOS = 4

# Intialize frame loader
frame_loader = data.FrameLoader(max_videos=MAX_VIDEOS)


with tf.Session() as sess:
    for image, target in frame_loader:
        output = sess.run(bottleneck_tensor, feed_dict={
            input_tensor: image.reshape((1,) + image.shape)
        })
        print(output)

#return train_op, one_hot_target_tensor, final_tensor
