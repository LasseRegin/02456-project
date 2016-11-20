#!/bin/sh

export K_FOLDS=5
#export K_FOLDS=2 # TODO: Remove this

# Run logistic regression
NUM_EPOCHS=500 LEARNING_RATE=1e-4 python logistic_classifier_cv.py
#K_FOLDS=2 NUM_EPOCHS=2 LEARNING_RATE=1e-4 python logistic_classifier_cv.py

# Run logistic regression on features from inception graph
NUM_EPOCHS=1000 LEARNING_RATE=1e-4 python logistic_features_classifier_cv.py
#K_FOLDS=2 NUM_EPOCHS=2 LEARNING_RATE=1e-4 python logistic_features_classifier_cv.py

# Run convolutional classifier
NUM_EPOCHS=200 LEARNING_RATE=1e-5 python conv_classifier_cv.py
#K_FOLDS=2 NUM_EPOCHS=2 LEARNING_RATE=1e-5 python conv_classifier_cv.py

# Run RNN classifier on features from inception graph
N_STEPS=15 KEEP_PROB=0.5 NUM_EPOCHS=500 LEARNING_RATE=1e-5 python rnn_features_classifier_cv.py
#K_FOLDS=2 N_STEPS=15 KEEP_PROB=0.5 NUM_EPOCHS=2 LEARNING_RATE=1e-5 python rnn_features_classifier_cv.py


# Evaluate runs
python eval_training.py simple-model-1
python eval_training.py simple-features-model-1
python eval_training.py conv-model-1
python eval_training.py rnn-features-model-1
