from __future__ import print_function, division, absolute_import

import os
import tensorflow as tf
from tensorflow.python.platform import gfile

import tensorflow.contrib.slim as slim

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

# Create final training operations
with tf.name_scope('targets'):
    one_hot_target_tensor = tf.placeholder(
        tf.float32, [None, MODEL_OUTPUT_SIZE]
    )

layer_name = 'final_training_ops'
with slim.arg_scope([slim.fully_connected],
                    weights_initializer=tf.truncated_normal_initializer(
                        stddev=0.001),
                    weights_regularizer=slim.l2_regularizer(0.0005),
                    scope=layer_name):
    logits = slim.fully_connected(bottleneck_tensor,
                                  MODEL_OUTPUT_SIZE,
                                  activation_fn=None)

final_tensor = tf.nn.softmax(logits, name=FINAL_TENSOR_NAME)

cross_entropy = slim.losses.softmax_cross_entropy(
    logits, one_hot_target_tensor, scope='cross_entropy')

optimizer = tf.train.AdamOptimizer(LEARNING_RATE)

with tf.name_scope('train'):
    train_op = slim.learning.create_train_op(cross_entropy, optimizer)



#return train_op, one_hot_target_tensor, final_tensor
