#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import random
import time
from collections import Counter

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.externals import joblib

def usage():
    sys.stderr.write('usage: python api_frequency.py data.csv output-model.pkl\n')
    sys.exit(2)

def _main():
    if len(sys.argv) != 3:
        usage()

    infile = sys.argv[1]
    outfile = sys.argv[2]

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

    # Create Multinomial Naive Bayes class
    # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
    clf = MultinomialNB()

    # Run training
    sys.stdout.write('Running training\n')
    t = time.time()
    clf.fit(train[:,1:len(train[0])-1].astype(np.float64), train[:,len(train[0])-1].astype(np.float64))
    sys.stdout.write('    Took {0} seconds\n'.format(str(time.time()-t)))

    # Run predictions
    sys.stdout.write('Running predictions\n')
    predicted = clf.predict(test[:,1:len(test[0])-1])
    accuracy = accuracy_score(test[:,len(test[0])-1].astype(np.float64), predicted)

    sys.stdout.write('\n')
    sys.stdout.write('Validation Accuracy: {0:.3}\n'.format(accuracy))

    # Dump model to file
    joblib.dump(clf, outfile)

if __name__ == '__main__':
    _main()
