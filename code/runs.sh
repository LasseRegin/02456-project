#!/bin/sh

# Run logistic regression
NUM_EPOCHS=200 LEARNING_RATE=1e-4 logistic_classifier.py

# Run logistic regression on features from inception graph
NUM_EPOCHS=200 LEARNING_RATE=1e-4 logistic_features_classifier.py



# Evaluate runs
python eval_training.py simple-model-1
python eval_training.py simple-features-model-1
