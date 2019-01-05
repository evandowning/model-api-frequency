import sys
import os
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def usage():
    print 'usage: python evaluation.py data.csv labels.txt output-model.pkl predictions.csv'
    sys.exit(2)

def _main():
    if len(sys.argv) != 5:
        usage()

    infile = sys.argv[1]
    label_fn = sys.argv[2]
    outfile = sys.argv[3]
    prediction_fn = sys.argv[4]

    # Create a map between malware family label and their integer representation
    labelMap = dict()
    with open(label_fn,'r') as fr:
        for e,line in enumerate(fr):
            line = line.strip('\n')
            labelMap[line] = e

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
    predicted = clf.predict(x[:,1:len(x[0])-1])
    accuracy = accuracy_score(x[:,len(x[0])-1].astype(int), predicted)

    print ''
    print 'Accuracy: {0:.3}'.format(accuracy)

    # Print predictions
    with open(prediction_fn,'w') as fw:
        fw.write('Hash,Label,Prediction\n')

        for e,p in enumerate(predicted):
            fw.write('{0},{1},{2}\n'.format(x[e][0], labelMap.keys()[labelMap.values().index(x[e][-1])], labelMap.keys()[labelMap.values().index(p)]))

if __name__ == '__main__':
    _main()
