#!/bin/sh

# Run logistic regression
NUM_EPOCHS=300 LEARNING_RATE=1e-4 python logistic_classifier.py

# Run logistic regression on features from inception graph
NUM_EPOCHS=1000 LEARNING_RATE=1e-4 python logistic_features_classifier.py

# Run convolutional classifier
NUM_EPOCHS=200 LEARNING_RATE=1e-5 python conv_classifier.py

# Run RNN classifier on features from inception graph
N_STEPS=15 KEEP_PROB=0.5 NUM_EPOCHS=500 LEARNING_RATE=1e-5 python rnn_features_classifier.py


# Evaluate runs
python eval_training.py simple-model-1
python eval_training.py simple-features-model-1
python eval_training.py conv-model-1
python eval_training.py rnn-features-model-1
