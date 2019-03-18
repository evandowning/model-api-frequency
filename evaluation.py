#!/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
import time

from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def usage():
    sys.stderr.write('usage: python evaluation.py data.csv labels.txt output-model.pkl predictions.csv\n')
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
    sys.stdout.write('Reading in data\n')
    t = time.time()
    data = pd.read_csv(infile,header=None)
    x = data.values
    sys.stdout.write('    Took {0} seconds\n'.format(str(time.time()-t)))

    # Load model
    clf = joblib.load(outfile)

    # Run predictions
    sys.stdout.write('Running predictions\n')
    predicted = clf.predict(x[:,1:len(x[0])-1])
    accuracy = accuracy_score(x[:,len(x[0])-1].astype(np.float64), predicted)

    sys.stdout.write('\n')
    sys.stdout.write('Accuracy: {0:.3}\n'.format(accuracy))

    # Print predictions
    with open(prediction_fn,'w') as fw:
        fw.write('Hash,Label,Prediction\n')

        for e,p in enumerate(predicted):
            fw.write('{0},{1},{2}\n'.format(x[e][0], list(labelMap.keys())[list(labelMap.values()).index(x[e][-1])], list(labelMap.keys())[list(labelMap.values()).index(p)]))

if __name__ == '__main__':
    _main()
