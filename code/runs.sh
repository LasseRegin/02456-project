#!/bin/sh

# Run logistic regression
NUM_EPOCHS=300 LEARNING_RATE=1e-4 python logistic_classifier.py

# Run logistic regression on features from inception graph
NUM_EPOCHS=300 LEARNING_RATE=1e-4 python logistic_features_classifier.py

# Run convolutional classifier
NUM_EPOCHS=200 LEARNING_RATE=1e-5 python conv_classifier.py


# Evaluate runs
python eval_training.py simple-model-1
python eval_training.py simple-features-model-1
python eval_training.py conv-model-1
