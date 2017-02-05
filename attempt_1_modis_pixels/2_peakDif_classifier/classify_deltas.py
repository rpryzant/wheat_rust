"""
same as the R script!! everything's predicted as 1!!


=== DESCRIPTION
Given some MODIS pixels and binary disease labels, this file trains
  some simple linear classifiers to try and seperate diseased from 
  healthy pixels

=== USAGE
python classify_deltas.py /Users/rapigan/Dropbox/school/ermon_lab/2_peakDif_classifier/data/diseased_thresholds/1/max_minus_min.csv 
"""

import numpy as np
import pandas as pd
import sys
from sklearn import svm
from sklearn import ensemble
import sklearn
import random
from itertools import chain, combinations







def get_subsets(i):
    for z in chain.from_iterable(combinations(i, r) for r in range(len(i)+1)):
        yield z


#  [5, 7]   ARE THE BEST

f = sys.argv[1]
data = pd.read_csv(f, header = 0)
#data = data.reindex(np.random.permutation(data.index))    # shuffle data

# if i standardize inputs, predictions all go to 1....wut
#data.iloc[:,1:] -= np.mean(data.iloc[:,1:].values, axis=0)
#data.iloc[:,1:] /= np.std(data.iloc[:,1:].values, axis=0)


Y = list(data.diseased)
N = len(Y)

best_subset = None
max_acc = -1

# try all feature subsets
#for subset in get_subsets(range(1, 8)):    # exclude label at (index 0)
#    print subset
#    subset = list(subset)
#    if not subset:
#        continue

# bands 5, 7 are most predictive (via feature selection)
X = [list(data.iloc[i].values) for i in range(len(data))] 



## train and test splits
split = N - (N/8)

X_train = X[:split]
X_test = X[split:]

Y_train = Y[:split]
Y_test = Y[split:]


## fit model
model = svm.SVC()
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)



####### analysis
def accuracy(Y1, Y2):
    return sum(1 if x == y else 0 for (x, y) in zip(Y1, Y2)) * 1.0 / len(Y1)


#acc = accuracy(Y_test, Y_pred)
#if acc > max_acc:
#    max_acc = acc
#    best_subset = subset




print 'MODEL:'
print Y_pred
print sklearn.metrics.classification_report(Y_test, Y_pred)
print 'acc: ', accuracy(Y_test, Y_pred)
print '========================================='


print 'RANDOM (predicted label frequencies):'
N1 = np.count_nonzero(Y_pred)
N0 = len(Y_pred) - N1
dummys = [1 if ( random.random() < (N1 * 1.0 / (N0 + N1)) ) else 0 for _ in range(len(Y_test))]
print sklearn.metrics.classification_report(Y_test, dummys)
print 'acc: ', accuracy(Y_test, dummys)
print '========================================='

common_label = 1 if N1 > N0 else 0
print 'ALL %ds:' % common_label
dummys = [common_label for _ in range(len(Y_test))]
print sklearn.metrics.classification_report(Y_test, dummys)
print 'acc: ', accuracy(Y_test, dummys)
print '========================================='

print 'RANDOM (global label frequencies):'
N0 = Y.count(0)
N1 = Y.count(1)
dummys = [1 if ( random.random() < (N1 * 1.0 / (N0 + N1)) ) else 0 for _ in range(len(Y_test))]
print sklearn.metrics.classification_report(Y_test, dummys)
print 'acc: ', accuracy(Y_test, dummys)
print '========================================='





print '========================================='
print '========================================='
print '========================================='
print '========================================='
