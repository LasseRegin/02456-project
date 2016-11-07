
import numpy as np
import tensorflow as tf

from network.base import Network

class ConvolutionalClassifier(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_variables(self):

        # Layer 1
        self.conv_1_W = tf.Variable(
            initial_value=tf.truncated_normal([5, 5, 3, 16], stddev=0.1),
            name='weights-conv-layer-1'
        )
        self.conv_1_b = tf.Variable(initial_value=tf.constant(0.1, shape=[16]), name='biases-layer-1')

        # Layer 2
        self.conv_2_W = tf.Variable(
            initial_value=tf.truncated_normal([5, 5, 16, 16], stddev=0.1),
            name='weights-conv-layer-2'
        )
        self.conv_2_b = tf.Variable(initial_value=tf.constant(0.1, shape=[16]), name='biases-layer-2')

        # Layer 3
        new_img_shape = np.ceil([self.input_shape[1] / 4, self.input_shape[2] / 4]).astype('int')
        self.W_3 = tf.Variable(
            initial_value=tf.truncated_normal(
                [np.prod(new_img_shape) * 16, 128],
                stddev=0.1
            ),
            name='weights-layer-3'
        )
        self.b_3 = tf.Variable(tf.constant(0.1, shape=[128]), name='biases-layer-3')

        # Layer 4
        self.W_4 = tf.Variable(
            initial_value=tf.truncated_normal(
                [128, self.target_shape[1]],
                stddev=0.1
            ),
            name='weights-layer-4'
        )
        self.b_4 = tf.Variable(tf.constant(0.1, shape=[self.target_shape[1]]), name='biases-layer-4')

    def init_network(self):

        # Layer 1 - Convolution
        conv_1 = tf.nn.conv2d(
            input=self.x,
            filter=self.conv_1_W,
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv-layer-1'
        )

        # Layer 1 - Non-linearity
        #relu_1 = tf.nn.relu(tf.nn.bias_add(bn_1, self.conv_1_b), name='relu-layer-1')
        relu_1 = tf.nn.relu(tf.nn.bias_add(conv_1, self.conv_1_b), name='relu-layer-1')

        # Layer 1 - Pooling
        pool_1 = tf.nn.max_pool(
            value=relu_1,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool-layer-1'
        )

        # Layer 2 - Convolution
        conv_2 = tf.nn.conv2d(
            input=pool_1,
            filter=self.conv_2_W,
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='conv-layer-2'
        )

        # Layer 2 - Non-linearity
        #relu_2 = tf.nn.relu(tf.nn.bias_add(bn_2, self.conv_2_b), name='relu-layer-2')
        relu_2 = tf.nn.relu(tf.nn.bias_add(conv_2, self.conv_2_b), name='relu-layer-2')

        # Layer 2 - Pooling
        pool_2 = tf.nn.max_pool(
            value=relu_2,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool-layer-2'
        )

        # Layer 3 - Reshaping
        pool_2_shape = pool_2.get_shape().as_list()
        reshape = tf.reshape(
            tensor=pool_2,
            shape=(-1, np.prod(pool_2_shape[1:]))
        )

        # Layer 3 - Fully connected
        a_3 = tf.matmul(reshape, self.W_3) + self.b_3
        z_3 = tf.nn.relu(a_3, name='relu-layer-3')

        # Layer 4 - Softmax
        a_4 = tf.matmul(z_3, self.W_4) + self.b_4
        z_4 = tf.nn.softmax(a_4)

        # Final output
        self.output = z_4

    def init_optimizer(self):
        # Minimize mean squared error
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.output, 1e-10, 1.0)), reduction_indices=[1]))

        # Gradient Descent
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.grads_and_vars = optimizer.compute_gradients(self.cost, [self.conv_1_W, self.conv_2_W, self.W_3, self.W_4])

        # Define optimization operation step
        self.optimizer_step = optimizer.minimize(self.cost)



# class ConvolutionalClassifier(Network):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#
#     def init_variables(self):
#
#         # Layer 1
#         self.conv_1_W = tf.Variable(
#             initial_value=tf.truncated_normal([5, 5, 3, 32], stddev=0.1),
#             name='weights-conv-layer-1'
#         )
#         self.conv_1_b = tf.Variable(initial_value=tf.zeros([32]), name='biases-layer-1')
#
#         # Layer 2
#         self.conv_2_W = tf.Variable(
#             initial_value=tf.truncated_normal([5, 5, 32, 64], stddev=0.1),
#             name='weights-conv-layer-2'
#         )
#         self.conv_2_b = tf.Variable(initial_value=tf.zeros([64]), name='biases-layer-2')
#
#         # Layer 3
#         new_img_shape = np.ceil([self.input_shape[1] / 4, self.input_shape[2] / 4]).astype('int')
#         self.W_3 = tf.Variable(
#             initial_value=tf.truncated_normal(
#                 [np.prod(new_img_shape) * 64, 512],
#                 stddev=0.1
#             ),
#             name='weights-layer-3'
#         )
#         self.b_3 = tf.Variable(tf.constant(0.1, shape=[512]), name='biases-layer-3')
#
#         # Layer 4
#         self.W_4 = tf.Variable(
#             initial_value=tf.truncated_normal(
#                 [512, self.target_shape[1]],
#                 stddev=0.1
#             ),
#             name='weights-layer-4'
#         )
#         self.b_4 = tf.Variable(tf.constant(0.1, shape=[self.target_shape[1]]), name='biases-layer-4')
#
#     def init_network(self):
#
#         # Layer 1 - Convolution
#         conv_1 = tf.nn.conv2d(
#             input=self.x,
#             filter=self.conv_1_W,
#             strides=[1, 1, 1, 1],
#             padding='SAME',
#             name='conv-layer-1'
#         )
#
#         # Layer 1 - Batch normalization
#         # mean_1, var_1 = tf.nn.moments(x=conv_1, axes=[0, 1, 2]) # 'Global' norm see API
#         batch_mean_1, batch_var_1 = tf.nn.moments(x=conv_1, axes=[0], keep_dims=False)
#         offset_1 = tf.Variable(tf.zeros([1]))
#         scale_1  = tf.Variable(tf.ones([1]))
#
#         bn_1 = tf.nn.batch_normalization(
#             x=conv_1,
#             mean=batch_mean_1,
#             variance=batch_var_1,
#             offset=offset_1,
#             scale=scale_1,
#             variance_epsilon=1e-3,
#             name='batch-norm-layer-1'
#         )
#
#         # Layer 1 - Non-linearity
#         relu_1 = tf.nn.relu(tf.nn.bias_add(bn_1, self.conv_1_b), name='relu-layer-1')
#
#         # Layer 1 - Pooling
#         pool_1 = tf.nn.max_pool(
#             value=relu_1,
#             ksize=[1, 2, 2, 1],
#             strides=[1, 2, 2, 1],
#             padding='SAME',
#             name='pool-layer-1'
#         )
#
#         # Layer 2 - Convolution
#         conv_2 = tf.nn.conv2d(
#             input=pool_1,
#             filter=self.conv_2_W,
#             strides=[1, 1, 1, 1],
#             padding='SAME',
#             name='conv-layer-2'
#         )
#
#         # Layer 2 - Batch normalization
#         # mean_1, var_1 = tf.nn.moments(x=conv_1, axes=[0, 1, 2]) # 'Global' norm see API
#         batch_mean_2, batch_var_2 = tf.nn.moments(x=conv_2, axes=[0], keep_dims=False)
#         offset_2 = tf.Variable(tf.zeros([1]))
#         scale_2  = tf.Variable(tf.ones([1]))
#
#         bn_2 = tf.nn.batch_normalization(
#             x=conv_2,
#             mean=batch_mean_2,
#             variance=batch_var_2,
#             offset=offset_2,
#             scale=scale_2,
#             variance_epsilon=1e-3,
#             name='batch-norm-layer-2'
#         )
#
#         # Layer 2 - Non-linearity
#         relu_2 = tf.nn.relu(tf.nn.bias_add(bn_2, self.conv_2_b), name='relu-layer-2')
#
#         # Layer 2 - Pooling
#         pool_2 = tf.nn.max_pool(
#             value=relu_2,
#             ksize=[1, 2, 2, 1],
#             strides=[1, 2, 2, 1],
#             padding='SAME',
#             name='pool-layer-2'
#         )
#
#         # Layer 3 - Reshaping
#         pool_2_shape = pool_2.get_shape().as_list()
#         reshape = tf.reshape(
#             tensor=pool_2,
#             shape=(-1, np.prod(pool_2_shape[1:]))
#         )
#
#         # Layer 3 - Fully connected
#         a_3 = tf.matmul(reshape, self.W_3) + self.b_3
#         z_3 = tf.nn.relu(a_3, name='relu-layer-3')
#
#         # Layer 4 - Softmax
#         a_4 = tf.matmul(z_3, self.W_4) + self.b_4
#         z_4 = tf.nn.softmax(a_4)
#
#         # Final output
#         self.output = z_4
#
#     def init_optimizer(self):
#         # Minimize mean squared error
#         self.cost = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(tf.clip_by_value(self.output, 1e-10, 1.0)), reduction_indices=[1]))
#
#         # Gradient Descent
#         optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
#
#         # Define optimization operation step
#         self.optimizer_step = optimizer.minimize(self.cost)
#
