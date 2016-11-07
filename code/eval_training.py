
import os
import data
import math
import utils
import network

import numpy as np
import tensorflow as tf


# Load training losses
lossTracker = utils.LossTracker.load(name='simple-model-1')

print(lossTracker.train_loss)
print(lossTracker.val_loss)
print(lossTracker.test_loss)
