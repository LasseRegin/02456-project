
import numpy as np
import tensorflow as tf

from network.base import Network

class LogisticClassifier(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_variables(self):
        total_pixels = np.prod(self.input_shape[1:])
        self.x_reshaped = tf.reshape(self.x, shape=[-1, total_pixels])

        # Weights for layer 1
        # NOTE: Gradients seems more stable when initializing W to zeros.
        #       This can only be used with logistic regression.
        #self.W_1 = tf.Variable(tf.random_normal([total_pixels, self.target_shape[1]], stddev=0.35), name='weights-layer-1')
        #self.W_1 = tf.Variable(tf.random_normal([total_pixels, self.target_shape[1]], stddev=0.10), name='weights-layer-1')
        self.W_1 = tf.Variable(tf.zeros([total_pixels, self.target_shape[1]]), name='weights-layer-1')
        self.b_1 = tf.Variable(tf.zeros([self.target_shape[1]]), name='biases-layer-1')

    def init_network(self):

        # Layer 1
        a_1 = tf.matmul(self.x_reshaped, self.W_1) + self.b_1
        z_1 = tf.nn.softmax(a_1)

        # Final output
        self.output = z_1

    def init_optimizer(self):
        # Minimize cross entropy
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.output, 1e-10, 1.0)), reduction_indices=[1]))

        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

        self.grads_and_vars = optimizer.compute_gradients(self.cost, [self.W_1])

        # Define optimization operation step
        self.optimizer_step = optimizer.minimize(self.cost)
