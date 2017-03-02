from sklearn import svm
import numpy as np
from src.training.evaluation import accuracy

class SVM():

    def __init__(self, config):
        if config.l2 > 0:
            self.model = svm.LinearSVC(penalty='l2', C=config.l2)
        elif config.l1 > 0:
            self.model = svm.LinearSVC(penalty='l1', C=config.l1)
        else:
            self.model = svm.SVC() # still regularizes

    def fit_and_predict(self, data, val_data, sess):
        x_train, y_train = zip(*data)
        x_train = [np.reshape(x, [-1]).tolist() for x in x_train] # concatenate errythang
        print len(x_train)
        print len(y_train)
        self.model.fit(x_train, y_train)

        x_val, y_val = zip(*val_data)
        x_val = [np.reshape(x, [-1]).tolist() for x in x_val] # concatenate errythang
        y_pred = self.model.predict(x_val)

        return -1, accuracy(y_pred, y_val), -1


