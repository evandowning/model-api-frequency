import sys
import os
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def usage():
    print 'usage: python evaluation.py data.csv output-model.pkl'
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

    # Load model
    clf = joblib.load(outfile)

    # Run predictions
    print 'Running predictions'
    predicted = clf.predict(x[:,:len(x[0])-1])
    accuracy = accuracy_score(x[:,len(x[0])-1], predicted)

    print ''
    print 'Accuracy: {0:.3}'.format(accuracy)

if __name__ == '__main__':
    _main()
