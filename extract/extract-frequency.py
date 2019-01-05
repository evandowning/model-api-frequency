import sys
import os
import shutil
import zipfile
import numpy as np
from collections import Counter
from multiprocessing import Pool

import dump2file
import bson2sequence

def extract(h,d,raw):
    # Dump file contents
    dump2file.dump(os.path.join(d,raw))

    # Uncompress zip file
    # From: https://stackoverflow.com/questions/3451111/unzipping-files-in-python
    zippath = os.path.join(d,'stuff.zip')
    with zipfile.ZipFile(zippath,'r') as zip_ref:
        zip_ref.extractall(d)

    # Parse bson files and extract data
    sequence = bson2sequence.extract(os.path.join(d,'logs'))

    # Clean up files
    for fn in os.listdir(d):
        # Don't remove raw file
        if fn == raw:
            continue

        path = os.path.join(d,fn)

        # If directory
        if os.path.isdir(path):
            shutil.rmtree(path)
        # If file
        else:
            os.remove(path)

    return h,sequence

def extract_wrapper(args):
    return extract(*args)

# Gets raw files and their base directory names
# NOTE: I assume there's been only one run of each sample
def getFiles(folder,sampleMap):
    # Get base directories
    dirs = os.listdir(folder)

    ignore_dir = ['logs']
    ignore_fn  = ['','dump.pcap','analysis.log','stuff.zip']

    # Get raw files
    for d in dirs:
        # Ignore samples we don't care about
        if d not in sampleMap.keys():
            continue

        for directory,dirname,files in os.walk(os.path.join(folder,d)):
            # Ignore directories
            base = os.path.basename(directory)
            if base in ignore_dir:
                continue

            for fn in files:
                # Ignore files
                if fn not in ignore_fn:
                    yield (d,directory,fn)

def usage():
    print 'usage: python extract-frequency.py /data/arsa/nvmtrace-cuckoo-data/output api.txt label.txt filtered.txt output.csv'
    sys.exit(2)

def _main():
    if len(sys.argv) != 6:
        usage()

    rawDir = sys.argv[1]
    api_file = sys.argv[2]
    label_file = sys.argv[3]
    samples_file = sys.argv[4]
    outfile = sys.argv[5]

    # If outfile already exists, remove it
    if os.path.exists(outfile):
        os.remove(outfile)

    # Get number of API calls
    a = 0
    apiMap = dict()
    with open(api_file, 'rb') as fr:
        for e,line in enumerate(fr):
            line = line.strip('\n')
            apiMap[line] = e
            a += 1

    # Get label integer values
    labelMap = dict()
    with open(label_file,'r') as fr:
        for e,line in enumerate(fr):
            line = line.strip('\n')
            labelMap[line] = e

    # Get sample labels
    sampleMap = dict()
    with open(samples_file,'r') as fr:
        for line in fr:
            line = line.strip('\n')
            h,c = line.split('\t')
            sampleMap[h] = labelMap[c]

    # Get raw files and their corresponding directory
    rv = getFiles(rawDir,sampleMap)

    # Construct args
    args = [(h,d,raw) for h,d,raw in rv]

    # Extract each raw data file
    pool = Pool(20)
    results = pool.imap_unordered(extract_wrapper, args)

    # Consolidate all features to single CSV file
    with open(outfile, 'a') as fa:
        for e,r in enumerate(results):
            h,seq = r

            # Replace API calls with their unique integer value
            # https://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array#3404089
            seq = np.array(seq)
            newseq = np.copy(seq)
            for k,v in apiMap.iteritems():
                newseq[seq==k] = v
            seq = newseq

            x = np.array([0]*a)

            # Use python function to give you counts of elements in numpy array
            cp = Counter(seq)

            # Create feature vector for frequency features
            sa = set(seq)

            for i in sa:
                x[int(i)] = cp[i]

            # Append data to CSV file
            fa.write('{0},'.format(h))
            fa.write('{0}'.format(','.join(map(str,x))))
            fa.write(',{0}\n'.format(sampleMap[h]))

            sys.stdout.write('Extracting data: {0}/{1}\r'.format(e+1,len(args)))
            sys.stdout.flush()

    pool.close()
    pool.join()
    sys.stdout.write('\n')
    sys.stdout.flush()

if __name__ == '__main__':
    _main()
