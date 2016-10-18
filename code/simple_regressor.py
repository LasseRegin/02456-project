
import os
import data
import utils

import numpy as np
import tensorflow as tf

SHOW_PLOT = 'SHOW_PLOT' in os.environ

# Training parameters
NUM_EPOCHS = 50
LEARNING_RATE = 1e-5

# Intialize frame loader
frame_loader = data.FrameLoader(shuffle=True)
height, width = frame_loader.data.target_height, frame_loader.data.target_width

# TODO: Add test and validation
frame_loader_train = utils.Minibatch(frame_iterator=frame_loader)


#
# Setup model
#

# tf Graph input
x = tf.placeholder(tf.float32, shape=[None, height, width], name='input-x')
y = tf.placeholder(tf.float32, shape=[None, 2], name='target-y')

total_pixels = height * width
x_reshaped = tf.reshape(x, shape=[-1, total_pixels])

# Construct model
W_1 = tf.Variable(tf.random_normal([total_pixels, 2], stddev=0.35), name='weights-layer-1')
b_1 = tf.Variable(tf.zeros([2]), name="biases-layer-1")

a_1 = tf.matmul(x_reshaped, W_1) + b_1
z_1 = tf.nn.relu(a_1)

#W_2 = tf.Variable(tf.random_normal([1024, 32], stddev=0.35), name='weights-layer-2')
#b_2 = tf.Variable(tf.zeros([32]), name="biases-layer-2")
#
#a_2 = tf.matmul(z_1, W_2) + b_2
#z_2 = tf.nn.relu(a_2)
#
#W_3 = tf.Variable(tf.random_normal([32, 2], stddev=0.35), name='weights-layer-3')
#b_3 = tf.Variable(tf.zeros([2]), name="biases-layer-3")
#
#a_3 = tf.matmul(z_2, W_3) + b_3
#z_3 = tf.nn.sigmoid(a_3)



#pred = tf.matmul(x_reshaped, W) + b # Linear
#pred = tf.nn.sigmoid(a_1) # Sigmoid
pred = z_1

# Minimize mean squared error
cost = tf.reduce_mean(tf.square(y - pred))

# Gradient Descent
#optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
optimizer_step = optimizer.minimize(cost)

# # Compute gradients
#grads = optimizer.compute_gradients(cost)
#
# # Clip gradients
# clipped_grads = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in grads]
# optimizer_step = optimizer.apply_gradients(clipped_grads)

# Initializing the variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    plot = utils.LossPlot(show=SHOW_PLOT)
    for epoch in range(0, NUM_EPOCHS):

        train_loss = 0.
        train_batches = 0
        for images, targets in frame_loader_train:
            # Run optimization operation
            images /= images.std() # TODO: do somewhere else

            _, loss, y_hat = sess.run([optimizer_step, cost, pred], feed_dict={
                x: images,
                y: targets
            })
            #print(y_hat)
            #print(targets)

            assert not (np.isnan(train_loss) or np.isinf(train_loss)), 'Loss returned NaN/Inf'

            train_loss += loss
            train_batches += 1
        train_loss /= train_batches

        plot.update(epoch=epoch, loss=train_loss)

        print('Epoch %d/%d' % (epoch+1, NUM_EPOCHS))
        print('Train loss: %g' % (train_loss))
