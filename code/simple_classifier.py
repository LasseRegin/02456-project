
import data
import utils

import lasagne
import theano
import theano.tensor as T

# Training parameters
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3


# Intialize frame loader
frame_loader = data.FrameLoader(filename='GOPR2477', downsample=4, found_only=True)
frame_loader = utils.BallPositionPoint(frame_iterator=frame_loader)
frame_loader = utils.ReshapeAndStandardize(frame_iterator=frame_loader)

# Get shapes
width = height = channels = None
for input, target in frame_loader:
    channels, height, width = input.shape
    break

frame_loader = utils.Validation(frame_iterator=frame_loader, test_fraction=0.33)

frame_loader_train = utils.Minibatch(frame_iterator=frame_loader.train, batch_size=20)
frame_loader_test  = utils.Minibatch(frame_iterator=frame_loader.test, batch_size=20)


# Intialize symbolic variables
input_var  = T.ftensor4('inputs')
target_var = T.fmatrix('targets')

# Build network
network = lasagne.layers.InputLayer(shape=(None, channels, height, width), input_var=input_var)

network = lasagne.layers.DenseLayer(
    incoming=network, num_units=2, nonlinearity=lasagne.nonlinearities.sigmoid
)

# print('lasagne.layers.get_output_shape(network)')
# print(lasagne.layers.get_output_shape(network))

# Setup loss functions
train_prediction = lasagne.layers.get_output(network)
test_prediction  = lasagne.layers.get_output(network, deterministic=True)
train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_var).mean()
test_loss  = lasagne.objectives.categorical_crossentropy(test_prediction,  target_var).mean()

# Setup update functions
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
    train_loss, params, learning_rate=LEARNING_RATE, momentum=0.9
)

# Setup training and validation function
train_fn = theano.function([input_var, target_var], train_loss, updates=updates)
val_fn   = theano.function([input_var, target_var], test_loss)

for epoch in range(0, NUM_EPOCHS):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0

    for inputs, targets in frame_loader_train:
        train_err += train_fn(inputs, targets)
        train_batches += 1
    train_err /= train_batches

    val_err = 0
    val_batches = 0
    for inputs, targets in frame_loader_test:
        val_err += val_fn(inputs, targets)
        val_batches += 1
    val_err /= val_batches

    print('Epoch %d/%d' % (epoch + 1, NUM_EPOCHS))
    print('\tTrain loss: %g' % (train_err))
    print('\tVal loss: %g' % (val_err))
    print('')




#
