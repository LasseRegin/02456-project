
import os
import csv
import numpy as np
import collections
import matplotlib.pyplot as plt
import scipy
import scipy.stats

# Taken from:
# http://stackoverflow.com/a/15034143/2538589
def mean_uncertainty(data, confidence=0.95):
    mean = np.mean(data)
    # Compute standard error of the mean
    se = scipy.stats.sem(data)

    # Compute uncertainty
    delta = se * scipy.stats.t._ppf((1 + confidence) / 2.0, len(data) - 1)
    return mean, delta



import utils

models = {
    'simple-model-1': collections.defaultdict(list),
    'simple-features-model-1': collections.defaultdict(list),
    'conv-model-1': collections.defaultdict(list),
    'rnn-features-model-1': collections.defaultdict(list),
}

for name, values in models.items():
    # Get errors
    error_tracker = utils.ErrorCalculations(name=name)
    RUNS_FOLDER = error_tracker.RUNS_FOLDER

    # Load errors
    with open(error_tracker.run_filename, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            line = list(map(float, line))
            values['test_loss'].append(line[0])
            values['TP'].append(line[1])
            values['FP'].append(line[2])
            values['TN'].append(line[3])
            values['FN'].append(line[4])
            values['top_0'].append(line[5])
            values['top_1'].append(line[6])
            values['top_2'].append(line[7])
            values['top_3'].append(line[8])


for name, values in models.items():
    for key, val in values.items():
        values[key] = np.array(val)

# Create box plot
labels = []
test_losses = []
accuracies_1 = []
accuracies_2 = []

name2disp = {
    'conv-model-1': 'Conv',
    'simple-model-1': 'Logistic',
    'simple-features-model-1': 'Logistic w. features',
    'rnn-features-model-1': 'RNN'
}

for name, values in sorted(models.items()):
    labels.append(name2disp[name])
    test_losses.append(values['test_loss'])
    accuracies_1.append(values['TN'] + values['top_0'])
    accuracies_2.append(values['TN'] + values['top_1'])

print('Tess losses')
for i, data in enumerate(test_losses):
    print(labels[i])
    mean, delta = mean_uncertainty(data)
    print('%.2f +/- %.2f' % (mean, delta))
print('')

print('Accuracies top 1')
for i, data in enumerate(accuracies_1):
    print(labels[i])
    mean, delta = mean_uncertainty(data)
    print('%.2f +/- %.2f' % (mean, delta))
print('')

print('Accuracies top 2')
for i, data in enumerate(accuracies_2):
    print(labels[i])
    mean, delta = mean_uncertainty(data)
    print('%.2f +/- %.2f' % (mean, delta))
print('')


plt.figure()
plt.boxplot(test_losses)
plt.xticks(range(1, len(labels) + 1), labels)
plt.savefig(os.path.join(RUNS_FOLDER, 'test-loss.pdf'), bbox_inches='tight')

plt.figure()
plt.boxplot(accuracies_1)
plt.xticks(range(1, len(labels) + 1), labels)
plt.savefig(os.path.join(RUNS_FOLDER, 'accuracies-1.pdf'), bbox_inches='tight')

plt.figure()
plt.boxplot(accuracies_2)
plt.xticks(range(1, len(labels) + 1), labels)
plt.savefig(os.path.join(RUNS_FOLDER, 'accuracies-2.pdf'), bbox_inches='tight')
#plt.show()
