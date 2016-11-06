
import os
import pickle

FILEPATH = os.path.dirname(os.path.abspath(__file__))

class LossTracker:
    RUNS_FOLDER = os.path.join(FILEPATH, 'runs')

    def __init__(self, name, num_epochs, verbose=False):
        self.name = name
        self.epoch = 0
        self.num_epochs = num_epochs
        self.verbose = verbose

        self.train_loss = []
        self.val_loss = []
        self.test_loss = None

    def addEpoch(self, train_loss, val_loss):
        self.epoch += 1
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)

        self._print('Epoch %d/%d' % (self.epoch, self.num_epochs))
        self._print('Train loss: %g' % (train_loss))
        self._print('Val loss: %g' % (val_loss))

    def addFinalTestLoss(self, test_loss):
        self.test_loss = test_loss
        self._print('')
        self._print('Final test loss: %g' % (test_loss))

    @classmethod
    def get_run_filename(cls, name):
        return os.path.join(cls.RUNS_FOLDER, '%s.pkl' % (name))

    @classmethod
    def load(cls, name):
        with open(cls.get_run_filename(name=name), 'rb') as f:
            return pickle.load(f)

    def save(self):
        if not os.path.isdir(self.RUNS_FOLDER):
            os.mkdir(self.RUNS_FOLDER)

        with open(self.get_run_filename(name=self.name), 'wb') as f:
            pickle.dump(self, f)

    def _print(self, text):
        if self.verbose:
            print(text)
