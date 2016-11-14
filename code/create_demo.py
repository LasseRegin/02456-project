
import os
import sys
import numpy as np
import imageio
from scipy.misc import imresize

import tensorflow as tf

import data
import utils
import network


# Intialize frame loader
frame_loader = data.FrameLoader(max_videos=4)
height, width = frame_loader.data.target_height, frame_loader.data.target_width
cells_x = frame_loader.cells_x
cells_y = frame_loader.cells_y

# Get demo video filename
filename = frame_loader.data.get_demo_video_filename()

# Initialize network
#nn = network.ConvolutionalClassifier(name='conv-model-1',
#                                     input_shape=(None, frame_loader.data.ORIGINAL_HEIGHT, frame_loader.data.ORIGINAL_WIDTH, 3),
#                                     target_shape=(None, cells_x * cells_y + 1))

nn = network.LogisticClassifier(name='simple-model-1',
                                input_shape=(None, height, width, 3),
                                target_shape=(None, cells_x * cells_y + 1),
                                verbose=True)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # Load saved model
    nn.load(sess)

    reader = imageio.get_reader(filename,  'ffmpeg')
    for i, frame in enumerate(reader):
        frame_resized = imresize(frame, size=(299, 299, 3))

        # Predict
        image_input = frame_resized - frame_resized.mean()

        image_input.resize((1,) + image_input.shape)
        prediction = nn.predict(session=sess, x=image_input).flatten()

        print(i)
        print(frame.shape)
        print(frame_resized.shape)
        print(prediction.shape)
        print('')

        # TODO:
        # 1) Predict using network  √
        # 2) Draw on HD image       -
        # 3) Save to demo video     -
