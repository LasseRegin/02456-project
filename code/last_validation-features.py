
import os
import sys
import json
import math
import numpy as np
import imageio
from scipy.misc import imresize

import tensorflow as tf
from tensorflow.python.platform import gfile

import data
import utils
import network


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

# Intialize frame loader
frame_loader = data.FeatureLoader(max_videos=4)
height, width = frame_loader.data.target_height, frame_loader.data.target_width
cells_x = frame_loader.cells_x
cells_y = frame_loader.cells_y

# frame_loader = data.FeatureLoader(max_videos=4)
# input_size = frame_loader.BOTTLENECK_TENSOR_SIZE
# cells_x = frame_loader.cells_x
# cells_y = frame_loader.cells_y

# Get demo video filename
filename = frame_loader.data.get_demo_video_filename()
json_filename = os.path.join(frame_loader.DATA_FOLDER, '267b2b632f95b150a8bbebd346ee0727_11000_1000_1.json')
with open(json_filename, 'r') as f:
    annotation = json.load(f)

balls = annotation['balls']
targets = []
for i in range(0, 1000):
    # Get ball info
    ball = balls.get(str(i), None)
    found = ball is not None

    if found:
        x = ball['x'] / frame_loader.data.ORIGINAL_WIDTH
        y = ball['y'] / frame_loader.data.ORIGINAL_HEIGHT
    else:
        x = None
        y = None

    y = ballPositionHeatMap(
        found=found,
        x=x,
        y=y,
        cells_x=cells_x,
        cells_y=cells_y
    )
    targets.append(y)
targets = np.asarray(targets)

# Initialize network
nn = network.LogisticClassifier(name='simple-features-model-1',
                                input_shape=(None, frame_loader.BOTTLENECK_TENSOR_SIZE),
                                target_shape=(None, cells_x * cells_y + 1),
                                verbose=True)
# Load images
reader = imageio.get_reader(filename,  'ffmpeg')
inputs = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    for i, frame in enumerate(reader):
        frame_resized = imresize(frame, size=(299, 299, 3))

        # Predict
        image_input = frame_resized - frame_resized.mean()
        #image_input /= image_input.std() # TODO: Tmp

        # Load frozen inception graph
        with gfile.FastGFile(frame_loader.MODEL_PATH, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            bottleneck_tensor, input_tensor = tf.import_graph_def(
                graph_def,
                name='',
                return_elements=[frame_loader.BOTTLENECK_TENSOR_NAME, frame_loader.INPUT_TENSOR_NAME]
            )

        image_features = sess.run(bottleneck_tensor, feed_dict={
            input_tensor: image_input.reshape((1,) + image_input.shape)
        })

        image_input = image_features.flatten().astype('float32')
        inputs.append(image_input)
inputs = np.asarray(inputs)


class TestDataIterator:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __iter__(self):
        for i in range(0, 20):
            idx_from = (i    ) * 50
            idx_to   = (i + 1) * 50
            yield self.inputs[idx_from:idx_to], self.targets[idx_from:idx_to]

data_iterator = TestDataIterator(inputs, targets)
error_tracker = utils.ErrorCalculations(name=nn.name)
with tf.Session(config=config) as sess:

    # Load saved model
    nn.load(sess)

    # Finally evaluate on test data
    test_loss = 0.
    test_batches = 0
    for images, targets in data_iterator:
        test_loss += nn.val_op(session=sess, x=images, y=targets)
        test_batches += 1
    test_loss /= test_batches
    error_tracker.evaluate(sess, nn, data_iterator, cells_x, cells_y, test_loss)

error_tracker.save()
