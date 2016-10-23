
import numpy as np
import tensorflow as tf

from network.base import Network

class SimpleRegressor(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_variables(self):
        total_pixels = np.prod(self.input_shape[1:])
        self.x_reshaped = tf.reshape(self.x, shape=[-1, total_pixels])

        # Weights for layer 1
        self.W_1 = tf.Variable(tf.random_normal([total_pixels, 1024], stddev=0.35), name='weights-layer-1')
        self.b_1 = tf.Variable(tf.zeros([1024]), name="biases-layer-1")

        # Weights for layer 2
        self.W_2 = tf.Variable(tf.random_normal([total_pixels, 2], stddev=0.35), name='weights-layer-2')
        self.b_2 = tf.Variable(tf.zeros([2]), name="biases-layer-2")

    def init_network(self):

        # Layer 1
        a_1 = tf.matmul(self.x_reshaped, self.W_1) + self.b_1
        z_1 = tf.nn.relu(a_1)

        # Layer 2
        a_2 = tf.matmul(z_1, self.W_2) + self.b_2

        # Final output
        self.output = a_2

    def init_optimizer(self):
        # Minimize mean squared error
        self.cost = tf.reduce_mean(tf.square(self.y - self.output))

        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Define optimization operation step
        self.optimizer_step = optimizer.minimize(self.cost)
