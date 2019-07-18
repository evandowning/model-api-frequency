#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import random
import time
import argparse

from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.externals import joblib

def _main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='input csv file', required=True)
    parser.add_argument('--ensemble_model', help='output ensemble model', required=True)

    group = parser.add_argument_group('rf', 'Random Forest')
    group.add_argument('--trees', help='number of trees', required=False, default=500)
    group.add_argument('--rf_model', help='output model filepath', required=False)

    group = parser.add_argument_group('nb', 'Naive Bayes')
    group.add_argument('--nb_model', help='output model filepath', required=False)

    group = parser.add_argument_group('knn', 'K-nearest neighbor')
    group.add_argument('--k', help='value of k', required=False, default=500)
    group.add_argument('--knn_model', help='output model filepath', required=False)

    group = parser.add_argument_group('svm', 'SVM')
    group.add_argument('--gamma', help='value of gamma', required=False, default=2)
    group.add_argument('--c', help='value of c', required=False, default=1)
    group.add_argument('--svm_model', help='output model filepath', required=False)

    group = parser.add_argument_group('mlp', 'MLP')
    group.add_argument('--hidden_layer_sizes', help='each integer is separated by underscore character. the ith element represents the number of neurons in the ith hidden layer. e.g., 10_10 is 2 layers, 10 nodes each', required=False, default='10_10')
    group.add_argument('--mlp_model', help='output model filepath', required=False)

    args = parser.parse_args()

    # Store arguments
    infile = args.csv
    outE = args.ensemble_model

    trees = int(args.trees)
    outRF = args.rf_model

    outNB = args.nb_model

    k = int(args.k)
    outK = args.knn_model

    gamma = int(args.gamma)
    c = int(args.c)
    outSVM = args.svm_model

    layers = args.hidden_layer_sizes
    outMLP = args.mlp_model

    # See if there are an odd number of models chosen
    count = list([outRF,outNB,outK,outSVM,outMLP]).count(None)
    if count % 2 != 0:
        sys.stderr.write('Error. Did not choose an odd number of models.\n')
        sys.exit(1)

    # Read in data
    sys.stdout.write('Reading in data\n')
    t = time.time()
    data = pd.read_csv(infile,header=None)
    x = data.values
    sys.stdout.write('    Took {0} seconds\n'.format(str(time.time()-t)))

    # Split dataset
    sys.stdout.write('Shuffling & organizing dataset\n')
    t = time.time()
    random.shuffle(x)
    thresh = int(len(x)*0.9)
    train = x[:thresh]
    test = x[thresh:]
    sys.stdout.write('    Took {0} seconds\n'.format(str(time.time()-t)))

    model = list()

    # Initialize models
    if outRF is not None:
        m = RandomForestClassifier(n_estimators=trees)
        model.append(('Random Forest',m,outRF))
    if outNB is not None:
        m = MultinomialNB()
        model.append(('Multinomial Naive Bayes',m,outNB))
    if outK is not None:
        m = KNeighborsClassifier(n_neighbors=k)
        model.append(('K-Nearest Neighbors',m,outK))
    if outSVM is not None:
        m = SGDClassifier()
        model.append(('Stochastic Gradient Descent',m,outSVM))
    if outMLP is not None:
        l = [int(e) for e in layers.split('_')]
        m = MLPClassifier(hidden_layer_sizes=l, activation='logistic', solver='adam', max_iter=1000)
        model.append(('Multi-layer Perceptron',m,outMLP))

    estimators = list()

    # Train models
    for n,m,outFN in model:
        sys.stdout.write('Training {0}...'.format(n))
        sys.stdout.flush()

        # Run training
        t = time.time()
        m.fit(train[:,1:len(train[0])-1].astype(np.float64), train[:,len(train[0])-1].astype(np.float64))
        sys.stdout.write('Done\n')
        sys.stdout.write('    Took {0} seconds\n'.format(str(time.time()-t)))

        # Create a dictionary of our models
        estimators.append((n,m))

        sys.stdout.write('Running predictions...')
        sys.stdout.flush()

        # Run predictions
        t = time.time()
        predicted = m.predict(test[:,1:len(test[0])-1])
        sys.stdout.write('Done\n')
        sys.stdout.write('    Took {0} seconds\n'.format(str(time.time()-t)))

        accuracy = accuracy_score(test[:,len(test[0])-1].astype(np.float64), predicted)

        sys.stdout.write('\n')
        sys.stdout.write('Validation Accuracy: {0:.4}\n'.format(accuracy))

        # Dump model to file
        joblib.dump(m, outFN)

        sys.stdout.write('========================\n')
        sys.stdout.write('========================\n')

    # From: https://towardsdatascience.com/ensemble-learning-using-scikit-learn-85c4531ff86a

    # Create voting classifier
    ensemble = VotingClassifier(estimators, voting='hard')

    sys.stdout.write('Training Ensemble...')
    sys.stdout.flush()

    # Fit model to training data
    t = time.time()
    ensemble.fit(train[:,1:len(train[0])-1].astype(np.float64), train[:,len(train[0])-1].astype(np.float64))
    sys.stdout.write('Done\n')
    sys.stdout.write('    Took {0} seconds\n'.format(str(time.time()-t)))

    # Dump model to file
    joblib.dump(m, outE)

    # Test our model on the test data
    score = ensemble.score(test[:,1:len(test[0])-1].astype(np.float64), test[:,len(test[0])-1].astype(np.float64))

    sys.stdout.write('Ensemble Accuracy: {0:.3}\n'.format(score))

if __name__ == '__main__':
    _main()
