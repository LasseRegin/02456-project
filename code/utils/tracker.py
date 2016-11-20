
import os
import csv
import pickle
import collections
import numpy as np

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



class ErrorCalculations:
    RUNS_FOLDER = os.path.join(FILEPATH, 'runs')

    def __init__(self, name):
        self.name = name
        self.run_filename = os.path.join(self.RUNS_FOLDER, '%s-cv.csv' % self.name)
        self.data = []

    def evaluate(self, session, nn, test_data_loader, cells_x, cells_y, test_loss):
        counts = 0
        FP = FN = TP = TN = 0
        last_idx = cells_x * cells_y
        top_k_counter = collections.Counter()
        for features, targets in test_data_loader:
            predictions = nn.predict(session=session, x=features)

            for prediction, target in zip(predictions, targets):

                # Get predicted index
                y_idx = target.argmax()
                y_hat_idx = prediction.argmax()

                if y_hat_idx == last_idx:
                    # Predicts there is no ball
                    if y_idx == last_idx:
                        # True negative
                        TN += 1
                    else:
                        # False negative
                        FN += 1
                else:
                    # Predicts there is a ball
                    if y_idx == last_idx:
                        # False positive
                        FP += 1
                    else:
                        TP += 1
                        # Get 2d coordinates
                        y_hat_row = y_hat_idx // cells_x
                        y_hat_col = y_hat_idx - y_hat_row
                        y_row = y_idx // cells_x
                        y_col = y_idx - y_row

                        # Compute 1-norm of vector diff
                        norm = int(np.linalg.norm(np.array([y_hat_row - y_row, y_hat_col - y_col]), ord=1))
                        top_k_counter[norm] += 1
                counts += 1

        # Compute percentages
        TP_percentage = TP / counts
        FP_percentage = FP / counts
        TN_percentage = TN / counts
        FN_percentage = FN / counts

        # Compute cumulative count
        top_k_percentages = {}
        total_count = sum(count for count in top_k_counter.values())
        if TP > 0:
            for k in range(0, max(top_k_counter.keys())):
                counts = 0
                for _k, count in top_k_counter.items():
                    if _k <= k: counts += count

                top_percentage = counts / total_count
                percentage = TP_percentage * top_percentage
                top_k_percentages[k] = percentage
        else:
            top_k_percentages = {k: 0.0 for k in range(0, 4)}

        # Calculate classification accuracy
        #TODO: when plotting

        # Save to data
        self.data.append([
            test_loss,
            TP_percentage,
            FP_percentage,
            TN_percentage,
            FN_percentage,
            top_k_percentages[0],
            top_k_percentages[1],
            top_k_percentages[2],
            top_k_percentages[3],
        ])

    def save(self):

        # Save to file
        with open(self.run_filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'test_loss',
                'TP',
                'FP',
                'TN',
                'FN',
                'top-0',
                'top-1',
                'top-2',
                'top-3',
            ])

            for row in self.data:
                writer.writerow(row)
