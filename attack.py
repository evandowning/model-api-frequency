import sys
import os
import math
import time
import pandas as pd
import numpy as np
import cPickle as pkl
from multiprocessing import Pool
from collections import Counter

from sklearn.externals import joblib

# Based on https://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers#11116960
def print_top10(clf, number_benign, number_malicious):
    """Prints features with the highest coefficient values, per class"""

    print 'Most informative features:'
    print 'API index, coefficient, # features in samples, avg. # features in samples'
    # Prints out most informative features for model for the "positive" class (i.e., malicious)
    features = np.argsort(clf.coef_[0])[::-1]
    for f in features[:10]:
        print '\tbenign:   \t{0}\t{1}\t{2}\t{3}'.format(f, clf.coef_[0][f], clf.feature_count_[0][f], clf.feature_count_[0][f]/float(number_benign))
        print '\tmalicious:\t{0}\t{1}\t{2}\t{3}'.format(f, clf.coef_[0][f], clf.feature_count_[1][f], clf.feature_count_[1][f]/float(number_malicious))
        print ''

    return [api for api in features if clf.feature_count_[0][api] > 0]

def usage():
    print 'usage: python attack.py sequences/ api.txt data.csv model.pkl output/'
    sys.exit(2)

def _main():
    if len(sys.argv) != 6:
        usage()

    # Get parameter
    sample_dir = sys.argv[1]
    api_file = sys.argv[2]
    infile = sys.argv[3]
    modelfn = sys.argv[4]
    output_dir = sys.argv[5]

    # If output_dir doesn't exist yet
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get number of API calls
    a = 0
    with open(api_file, 'rb') as fr:
        for e,line in enumerate(fr):
            line = line.strip('\n')
            a += 1

    # Read in data
    print 'Reading in data'
    t = time.time()
    data = pd.read_csv(infile)
    x = data.values
    print '    Took {0} seconds'.format(str(time.time()-t))

    # Count number of benign samples
    y = x[:,len(x[0])-1]
    c = Counter(y)
    print c
    number_benign = c[0]
    number_malicious = c[1]

    # Load model
    print 'Loading model'
    clf = joblib.load(modelfn)

    # Print top most important features for benign samples
    attack_features = print_top10(clf, number_benign, number_malicious)

    # Read in metadata
    metadata_fn = os.path.join(sample_dir,'metadata.pkl')
    with open(metadata_fn,'rb') as fr:
        # Window Size
        windowSize = pkl.load(fr)
        # Number of samples per label
        labelCount = pkl.load(fr)
        # Number of samples per data file (so we can determine folds properly)
        fileMap = pkl.load(fr)

    sumToBenign = 0
    totalMalicious = 0
    reduceCalls = Counter()

    # Get malware files to find attacks for
    for root, dirs, files in os.walk(sample_dir):
        for filename in files[:100]:
            # Ignore metadata
            if filename == 'metadata.pkl':
                continue

            # Read in sequence data
            fn = os.path.join(root,filename)

            # If file is empty
            if os.stat(fn).st_size == 0:
                continue

            print '================================='
            print '================================='

            # Get chunk number
            s = os.path.basename(fn)
            count = fileMap[s[:-4]]

            seq = np.array([])
            
            # Read sequence
            with open(fn, 'rb') as fr:
                for i in range(count):
                    n,l = pkl.load(fr)
                    if len(seq) == 0:
                        seq = n
                    else:
                        seq = np.append(seq,n)

            x = np.array([0]*a)

            # Use python function to give you counts of elements in numpy array
            cp = Counter(seq)

            # Create feature vector for frequency features
            sa = set(seq)

            # Remove 0's from feature vector (these are padding integers)
            sa -= {0}
            # -1 because 0 is used as a padding character for sequences
            for i in sa:
                x[i-1] = cp[i]

            # For classes 'benign' and 'malicious'
            if l > 0:
                l = 1

            # If benign, ignore
            if l == 0:
                continue

            totalMalicious += 1

            #TODO - print out original prediction for this sample

            # Run predictions
            print 'Running predictions'
            predicted = clf.predict([x])[0]
            #accuracy = accuracy_score([x], predicted)

            print ''
            print '\tPredictions: {0}'.format(predicted)
            #print 'Validation Accuracy: {0:.3}'.format(accuracy)

            prob = clf.predict_proba([x])[0]
            print '\tProbability estimate: {0}'.format(prob)

            #TODO - attack the counts of this malware's features

            xprime = np.copy(x)

            #TODO
            special_reduce = [4137, 4673, 1928, 836, 3107, 1603, 1591, 2217, 2035, 1592, 1616, 3164]

            # http://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes
            print 'size of attack_features: {0}'.format(len(attack_features))
#           for f in attack_features[:35]:
            for f in attack_features:

                c = clf.feature_count_[0][f]/float(number_benign)
                # Change every feature
#               if x[f] != c:
#                   x[f] = math.ceil(c)

                # Only increase features (except for Sleep which we can decrease)
                if (xprime[f] < c) or (f in special_reduce):
                    xprime[f] = math.ceil(c)
                else:
                    print f,xprime[f],c

            # call predict on new data and see if it now classifies as benign

            # Run predictions
            print 'Running predictions'
            predicted = clf.predict([xprime])[0]

            print ''
            print '\tPredictions: {0}'.format(predicted)

            prob = clf.predict_proba([xprime])[0]
            print '\tProbability estimate: {0}'.format(prob)

            # If still malicious, try to find calls in common to reduce
            if predicted == 1:
                for f in attack_features:
                    c = clf.feature_count_[0][f]/float(number_benign)
                    # Only increase features (except for Sleep, which we can decrease)
                    if (x[f] < c) or (f in special_reduce):
                        continue
                    else:
                        reduceCalls[f] += 1

            if predicted == 0:
                sumToBenign += 1

    print sumToBenign,totalMalicious
    print '% Changed from malicious to benign: ', float(sumToBenign)/totalMalicious

    print reduceCalls

if __name__ == '__main__':
    _main()
