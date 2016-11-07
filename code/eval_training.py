
import os
import sys

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import utils

MODEL_NAME = 'simple-model-1'
if len(sys.argv) == 1:
    print('No model name provided as argument. Using default value of \"%s\"' % (MODEL_NAME))
else:
    MODEL_NAME = sys.argv[1]

print('Creating plot for model %s' % (MODEL_NAME))

# Load training losses
lossTracker = utils.LossTracker.load(name=MODEL_NAME)

plt.plot(lossTracker.train_loss, color='IndianRed', label='Train')
plt.plot(lossTracker.val_loss,   color='SteelBlue', label='Val')
plt.legend()

if lossTracker.test_loss is not None:
    plt.title('Final test error %g' % (lossTracker.test_loss))
else:
    plt.title('Final test error not available')

# Save plot
plt.savefig(os.path.join(lossTracker.RUNS_FOLDER, '%s.png') % (MODEL_NAME), bbox_inches='tight')
plt.savefig(os.path.join(lossTracker.RUNS_FOLDER, '%s.pdf') % (MODEL_NAME), bbox_inches='tight')
plt.show()
