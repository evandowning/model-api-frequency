import sys
import os
import math
import numpy as np
import cPickle as pkl
from subprocess import call
import itertools
from multiprocessing import Pool

from sklearn import tree
from sklearn.tree import _tree
from sklearn.externals import joblib

# NOTE: to defeat a random forest, we must defeat a majority
#       of the decision trees in that forest:
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier.predict

#TODO - make sure this works for any amount of successes and failure paths (i.e., how do we change 'x' if we have to go "backwards"?)
# Finds an attack on multiple trees simultaneously
# http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#
# https://stackoverflow.com/questions/48880557/print-the-decision-path-of-a-specific-sample-in-a-random-forest-classifier
def find_attack(trees, comb, dt_index, benign_paths, x, xorig, api_list):
#   print '-------------------'
#   print 'Attacking tree: {0}'.format(dt_index)

    # Get tree which belongs to this index
    index = comb[dt_index]
    estimator = trees[index]

    # Get benign paths which belong to this tree
    bpath = benign_paths[dt_index]

    # Get decision path for this sample for this tree
    node_indicator = estimator.decision_path([x])

    # Get leaves
    leave_id = estimator.apply([x])

    # Get nodes in decision path for this tree
    node_index = node_indicator.indices[node_indicator.indptr[0]:
                            node_indicator.indptr[1]]

    decision_path = list()

    # Construct decision path via directions (to compare to benign paths)
    features = estimator.tree_.feature
    value = estimator.tree_.value
    prev = None
    for n in node_index:
        # Figure out which direction this was traversed in
        if prev != None:
            # If left node
            if n == estimator.tree_.children_left[prev]:
                decision_path.append('left')
            else:
                decision_path.append('right')

        prev = n

        # If a leaf node is reached, a decision is made
        if leave_id[0] == n:
#           print 'class: ', str(np.argmax(value[n]))
            break

        # Append node
        decision_path.append(str(api_list[features[n]]))

#   print decision_path

    xcopy = None
    # For each benign path, see if there is a way to it from the decision path
    for bp in bpath:
#       print bp

        # Make a backup of the current 'x'
        xcopy = np.copy(x)

        answer = 'YES'
        bprev = None
        for d,b in itertools.izip_longest(decision_path,bp):
            # If these disagree with each other, check direction
            if d != b:
                # If it's a left direction, check to see if this isn't a feature in our sample (if it is, we can't use this attack)
                if b == 'left':
                    # Determine API index of this call
                    ni = int(bprev.split(' ')[0]) - 1
                    if x[ni] == 1:
                        answer = 'NO'
                        break

            bprev = b
#       print answer

        # If we've found an attack, change 'x' and move onto the next tree
        if answer == 'YES':
            bprev = None
            for b in bp:
                # "Add" API call to 'x'
                if b == 'right':
                    # Determine API index of this call
                    ni = int(bprev.split(' ')[0]) - 1
#                   if x[ni] != 1:
#                       print 'changing index: x[{0}]'.format(ni)
                    x[ni] = 1

                bprev = b

            # If we have no more trees to search through, we're done :)
            if dt_index+1 == len(comb):
                return 'success',x

            # Move onto next tree
            return find_attack(trees, comb, dt_index+1, benign_paths, x, xorig, api_list)

        # If we have to loop to try the next path, reset 'x'
        x = np.copy(xcopy)

    return 'failure',None

# Produces attack for each sample
def attack_all(fn, clf, benign_paths, num_trees, half, a, api_list, output_dir):
    # Read sequence
    with open(fn, 'rb') as fr:
        s,l = pkl.load(fr)

    # If sample is benign, ignore it
    if l == 0:
        return None
    # Else, change label to be malicious (instead of malware class)
    else:
        l = 1

    x = np.array([0]*a)
    # Deduplicate sequence integers
    s = set(s)

    # Remove 0's from feature vector (these are padding integers)
    s -= {0}

    # Create feature vector for existence
    # -1 because 0 is used as a padding character for sequences
    for i in s:
        x[i-1] = 1

    # Evaluate features
    pred = clf.predict([x])[0]

    if pred == 0:
#       print '    Already classified as benign'
        return None

    # For each combination of trees
    attack = None
    for comb in list(itertools.combinations(range(num_trees),half)):
#       print 'trying to attack combination: {0}'.format(str(comb))

        # Find an attack
        rv,attack = find_attack(clf.estimators_, comb, 0, benign_paths, np.copy(x), x, api_list)
#       print 'final: ', rv

        # If attack is found, we're done with this sample
        if attack is not None:
            break

    # If attack wasn't found
    if attack is None:
        return None

#   print '    Final: ', attack
#   print '    Added API calls: ', [i for i in range(len(x)) if attack[i] != x[i]]

    # Check the new label
#   print '    New predict: ', clf.predict([attack])[0]

    # Create output filename
    base = os.path.basename(fn)
    outfn = os.path.join(output_dir,base[:-4])

#   print '    Writing to {0}'.format(outfn)

    # Record added API call indices (changed to match the numbers in syscall sequence)
    with open(outfn, 'w') as fw:
        fw.write(str([i+1 for i in range(len(x)) if attack[i] != x[i]]))

    # Return statistics about how many api calls were added
    return sum([i for i in x if i == 1]), sum([i for i in attack if i == 1])

def attack_all_wrapper(args):
    return attack_all(*args)

# Prints out decision tree logic in if-else statement form
def recursive_print(left, right, threshold, features, node, value, depth=0):
    indent = "\t" * depth
    if(threshold[node] != -2):
        print indent, "if ( " + str(features[node]) + " <= " + str(threshold[node])  + " class:" + str(np.argmax(value[node])) +  "  ) { " 
        if left[node] != -1:
            recursive_print (left, right, threshold, features, left[node], value, depth+1)
            print indent, '} else { '
            if right[node] != -1:
                recursive_print (left, right, threshold, features, right[node], value, depth+1)
            print indent, ' } '
    else:
        print indent,"return "  + str(value[node])

# Print decision tree
def print_tree(tree_in_clf, api_list, fn, outfn):
    # Print tree logic to stdout
    # https://www.kdnuggets.com/2017/05/simplifying-decision-tree-interpretation-decision-rules-python.html
#   recursive_print(left, right, tree_in_clf.tree_.threshold, tree_in_clf.tree_.feature, node, tree_in_clf.tree_.value)

    print 'Writing tree to {0} and {1}'.format(fn, outfn)

    # https://stats.stackexchange.com/questions/118016/how-can-you-print-the-decision-tree-of-a-randomforestclassifier
    # Write tree information to dot file
    with open(fn, 'w') as fw:
        tree.export_graphviz(tree_in_clf, out_file = fw, class_names = ['benign', 'malicious'], feature_names = api_list)

    # Convert dot file to png file
    call(['dot', '-Tpng', fn, '-o', outfn, '-Gdpi=300'])

# Pythonic way of traversing tree & keeping track of path
# Based on https://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
def get_paths(t,start):
    # Initialize path
    queue = [(start, [start])]

    while queue:
        # Get current node/path
        (node, path) = queue.pop(0)

        # Get left and right children of node
        left = t.children_left[node]
        right = t.children_right[node]

        # If leaf node, yield path
        if (left == -1) and (right == -1):
            yield path
        else:
            # If a left child exists
            if left != -1:
                queue.append((left, path + ['left',left]))
            # If a right child exists
            if right != -1:
                queue.append((right, path + ['right',right]))

# Creates ruleset of single tree
def parse_tree(tree_,feature_names):
    threshold = tree_.threshold
    features = tree_.feature

    #TODO - number of samples and values do not match
    value = tree_.value
    samples = tree_.n_node_samples
#   print value, samples

    # Get all paths in decision tree
    paths = list(get_paths(tree_,0))

    print '    Number of paths: ', str(len(paths))

    rules = list()

    # Extract rulesets for benign paths
    for p in paths:
        # Get last node in path (the class)
        c = p[-1]

        # If the class is benign
        if np.argmax(value[c]) == 0:
            f = list()

            for n in p:
                if n == 'left' or n == 'right':
                    f.append(n)
                    continue

                if threshold[n] != -2:
                    f.append(str(feature_names[features[n]]))

            rules.append(f)

    return rules

# Returns list of rulesets for benign paths
def create_rules(clf, api_list):
    rules = list()

    # Iterate over each tree in random forest
    for e, tree_in_clf in enumerate(clf.estimators_):
        print 'Scanning tree {0}'.format(e)

        # Create ruleset for tree
        rv = parse_tree(tree_in_clf.tree_,api_list)
        rules.append(rv)

#       # Print tree to file
#       fn = 'tree_' + str(e) + '.dot' 
#       outfn = 'tree_' + str(e) + '.png'
#       print_tree(tree_in_clf, api_list, fn, outfn)

    return rules

def usage():
    print 'usage: python attack.py sequences/ api.txt model.pkl output/'
    sys.exit(2)

def _main():
    if len(sys.argv) != 5:
        usage()

    # Get parameter
    sample_dir = sys.argv[1]
    api_file = sys.argv[2]
    modelfn = sys.argv[3]
    output_dir = sys.argv[4]

    # If output_dir doesn't exist yet
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Get number of API calls
    a = 0
    api_list = list()
    with open(api_file, 'rb') as fr:
        for e,line in enumerate(fr):
            line = line.strip('\n')
            api_list.append('{0} {1}'.format(e+1,line))
            a += 1

    # Load model
    print 'Loading model'
    clf = joblib.load(modelfn)

    # Extract benign paths for each tree
    benign_paths = create_rules(clf, api_list)


    # Calculate combination of trees which would
    # cause random forest to misclassify
    num_trees = len(clf.estimators_)
    # If evenly divisible
    if num_trees % 2 == 0:
        half = num_trees/2 + 1
    else:
        half = int(math.ceil(num_trees/2.0))

    print ''
    print 'Must find attacks for {0}/{1} trees'.format(half,num_trees)

    args = list()

    # Get malware files to find attacks for
    for root, dirs, files in os.walk(sample_dir):
        for filename in files:
            # Ignore metadata
            if filename == 'metadata.pkl':
                continue

            # Read in sequence data
            fn = os.path.join(root,filename)

            # If file is empty
            if os.stat(fn).st_size == 0:
                continue

            args.append((fn,clf,benign_paths,num_trees,half,a,api_list,output_dir))

    statSum = 0
    statTotal = 0

    pool = Pool(20)
    results = pool.imap_unordered(attack_all_wrapper, args)
    for e,r in enumerate(results):
        sys.stdout.write('Finding attacks: {0}/{1}\r'.format(e+1,len(args)))
        sys.stdout.flush()

        if r is not None:
            before,after = r
            statSum += ((float(after) / before) - 1)
            statTotal += 1

    pool.close()
    pool.join()

    sys.stdout.write('\n')
    sys.stdout.flush()

    # Print statistics
    print 'Avg. added % of api calls: {0}'.format(statSum * 100.0 / float(statTotal))

if __name__ == '__main__':
    _main()
