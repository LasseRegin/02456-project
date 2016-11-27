
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
    fps = 30000 / 1001
    writer = imageio.get_writer('demo.mp4', 'ffmpeg', fps=fps)
    for i, frame in enumerate(reader):
        frame_resized = imresize(frame, size=(299, 299, 3))

        # Predict
        image_input = frame_resized - frame_resized.mean()

        image_input.resize((1,) + image_input.shape)
        prediction = nn.predict(session=sess, x=image_input).flatten()

        # If we predict the ball being there -> draw the probabilities on frame
        #if prediction.argmax() < len(prediction) - 1:
        if True: #TODO:
            heat_map = prediction[:-1]

            # TODO: remove this
            heat_map /= heat_map.max()

            heat_map.resize((cells_y, cells_x))




            # Create full-size filter
            heat_filter = np.zeros(frame.shape)
            indices_x = int(frame.shape[1] // cells_x)
            indices_y = int(frame.shape[0] // cells_y)
            for i in range(0, cells_y):
                for j in range(0, cells_x):
                    idx_from_x =  j      * indices_x
                    idx_to_x   = (j + 1) * indices_x
                    idx_from_y =  i      * indices_y
                    idx_to_y   = (i + 1) * indices_y

                    heat_filter[idx_from_y:idx_to_y, idx_from_x:idx_to_x] = heat_map[i,j]

            # Apply filter to frame
            frame = frame * heat_filter
            frame = frame.astype('uint8')

        writer.append_data(frame)
    writer.close()

        # TODO:
        # 1) Predict using network  √
        # 2) Draw on HD image       -
        # 3) Save to demo video     -
