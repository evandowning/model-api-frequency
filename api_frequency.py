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
    print 'usage: python api_frequency.py data.csv output-model.pkl'
    sys.exit(2)

def _main():
    if len(sys.argv) != 3:
        usage()

    infile = sys.argv[1]
    outfile = sys.argv[2]

    # Read in data
    print 'Reading in data'
    t = time.time()
    data = pd.read_csv(infile)
    x = data.values
    print '    Took {0} seconds'.format(str(time.time()-t))

    # Split dataset
    print 'Shuffling & organizing dataset'
    t = time.time()
    random.shuffle(x)
    thresh = int(len(x)*0.9)
    train = x[:thresh]
    test = x[thresh:]
    print '    Took {0} seconds'.format(str(time.time()-t))

    # Create Multinomial Naive Bayes class
    # http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
    clf = MultinomialNB()

    # Run training
    print 'Running training'
    t = time.time()
    clf.fit(train[:,1:len(train[0])-1].astype(np.float64), train[:,len(train[0])-1].astype(np.float64))
    print '    Took {0} seconds'.format(str(time.time()-t))

    # Run predictions
    print 'Running predictions'
    predicted = clf.predict(test[:,1:len(test[0])-1])
    accuracy = accuracy_score(test[:,len(test[0])-1].astype(np.float64), predicted)

    print ''
    print 'Validation Accuracy: {0:.3}'.format(accuracy)

    # Dump model to file
    joblib.dump(clf, outfile)

if __name__ == '__main__':
    _main()
