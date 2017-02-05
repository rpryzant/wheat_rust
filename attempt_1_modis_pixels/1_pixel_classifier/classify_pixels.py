"""
same as the R script!! everything's predicted as 1!!


=== DESCRIPTION
Given some MODIS pixels and binary disease labels, this file trains
  some simple linear classifiers to try and seperate diseased from 
  healthy pixels

=== USAGE
python classify_pixels.py pixels.csv
"""

import numpy as np
import pandas as pd
import sys
from sklearn import svm
import random

f = sys.argv[1]
data = pd.read_csv(f, header = 0)
data.reindex(np.random.permutation(data.index))    # shuffle data


Y = list(data.diseased)
# lop off metadata and last feature (TODO: RERUN PIXEL HARVEST, I MISSED BAND 7 :((  )
X = [list(data.iloc[i].values)[5:-1] for i in range(len(data))]
N = len(Y)


print Y.count(0)
print Y.count(1)


## train and test splits
split = N - (N/8)

X_train = X[:split]
X_test = X[split:]

Y_train = Y[:split]
Y_test = Y[split:]


## fit model
model = svm.SVC()
model.fit(X_train, Y_train)


# broken??
print model.predict(X_test)





