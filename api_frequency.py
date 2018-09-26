import sys
import os
import pandas as pd
import numpy as np
import random
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
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

    #TODO - use multinomial nb - http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB
    # Create Random Forest
    clf = RandomForestClassifier(n_estimators=10)

    # Run training
    print 'Running training'
    t = time.time()
    clf.fit(train[:,:len(train[0])-1], train[:,len(train[0])-1])
    print '    Took {0} seconds'.format(str(time.time()-t))

    # Print out "n most important features"
    # https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
    n = 10
    imp = clf.feature_importances_
    index = imp.argsort()[-n:][::-1]
    # Print important API calls
    for i in index:
        print 'Call: {0}    Importance: {1}'.format(i,imp[i])

    # Run predictions
    print 'Running predictions'
    predicted = clf.predict(test[:,:len(test[0])-1])
    accuracy = accuracy_score(test[:,len(test[0])-1], predicted)

    print ''
    print 'Validation Accuracy: {0:.3}'.format(accuracy)

    # Dump model to file
    joblib.dump(clf, outfile)

if __name__ == '__main__':
    _main()
