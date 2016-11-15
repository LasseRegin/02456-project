
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

from network.base import Network

class RNNClassifier(Network):
    def __init__(self, n_steps, dropout=0.0, **kwargs):
        self.n_steps = n_steps
        self.dropout = dropout
        super().__init__(**kwargs)

    def init_variables(self):

        # Permuting batch_size and n_steps
        self.x_reshaped = tf.transpose(self.x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, input_size)
        self.x_reshaped = tf.reshape(self.x_reshaped, (-1, self.input_shape[-1]))
        # Split to get a list of 'n_steps' tensors of shape (batch_size, input_size)
        self.x_reshaped = tf.split(0, self.n_steps, self.x_reshaped)


        # Weights for layer 1
        self.W_1 = tf.Variable(tf.random_normal([128, self.target_shape[1]]), name='weights-layer-1')
        self.b_1 = tf.Variable(tf.random_normal([self.target_shape[1]]), name='biases-layer-1')

    def init_network(self):

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(128, forget_bias=1.0)

        # Add dropout
        if self.dropout > 0.0:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=self.dropout
            )


        #lstm = rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=False)
        #stacked_lstm = rnn_cell.MultiRNNCell([lstm] * number_of_layers, state_is_tuple=False)
        #stacked_lstm = rnn_cell.MultiRNNCell([lstm_cell] * 4)

        # Get lstm cell output
        #outputs, states = rnn.rnn(stacked_lstm, self.x_reshaped, dtype=tf.float32)
        outputs, states = rnn.rnn(lstm_cell, self.x_reshaped, dtype=tf.float32)
        #outputs, states = rnn.rnn(cell=lstm_cell, inputs=self.x_reshaped, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        a_1 = tf.matmul(outputs[-1], self.W_1) + self.b_1
        z_1 = tf.nn.softmax(a_1)

        # Final output
        self.output = z_1


    def init_optimizer(self):
        # Minimize cross entropy
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.output, 1e-10, 1.0)), reduction_indices=[1]))

        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        # Define optimization operation step
        self.optimizer_step = optimizer.minimize(self.cost)
