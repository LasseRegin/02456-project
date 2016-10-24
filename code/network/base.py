
import os
import numpy as np
import tensorflow as tf

FILEPATH = os.path.dirname(os.path.abspath(__file__))


class Network:
    MODELS_FOLDER = os.path.join(FILEPATH, 'models')

    def __init__(self, name, input_shape, target_shape, learning_rate=1e-3,
                 verbose=False, **kwargs):
        # Name is used for saving and loading the model
        self.name = name
        self.verbose = verbose

        # Shapes
        self.input_shape = input_shape
        self.target_shape = target_shape

        # Hyperparameters
        self.learning_rate = learning_rate

        # Initialize network
        self._print('Setting up network..')
        self.init_placeholders()
        self.init_variables()
        self.init_network()
        self.init_optimizer()

        # Setup operation for initializing the variables
        self.init_op = tf.initialize_all_variables()

        # Add ops to save and restore all the variables
        self.saver = tf.train.Saver()

    def init(self, session):
        session.run(self.init_op)

    def get_model_filepath(self):
        return os.path.join(self.MODELS_FOLDER, '%s.ckpt' % (self.name))

    def load(self, session):
        filepath = self.get_model_filepath()
        self.saver.restore(session, filepath)
        self._print('Loaded model from %s' % (filepath))

    def save(self, session):
        # Make models folder
        if not os.path.isdir(self.MODELS_FOLDER):
            os.mkdir(self.MODELS_FOLDER)

        # Save the variables to disk.
        save_path = self.saver.save(session, self.get_model_filepath())
        self._print("Saved model to %s" % (save_path))


    def init_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=self.input_shape,  name='input-x')
        self.y = tf.placeholder(tf.float32, shape=self.target_shape, name='target-y')

    def init_variables(self):
        raise NotImplementedError()

    def init_network(self):
        raise NotImplementedError()

    def init_optimizer(self):
        raise NotImplementedError()

    def _print(self, message):
        if self.verbose:
            print(message)

    def predict(self, session, x):
        return session.run(self.output, feed_dict={
            self.x: x
        })

    def train_op(self, session, x, y):
        return self.compute_loss(session, x, y, optimize=True)

    def val_op(self, session, x, y):
        return self.compute_loss(session, x, y, optimize=False)

    def compute_loss(self, session, x, y, optimize):
        if optimize:
            # With optimization step
            _, loss = session.run([self.optimizer_step, self.cost], feed_dict={
                self.x: x,
                self.y: y
            })
        else:
            # Without optimization step
            loss = session.run(self.cost, feed_dict={
                self.x: x,
                self.y: y
            })

        assert not (np.isnan(loss) or np.isinf(loss)), 'Loss returned NaN/Inf'

        return loss
