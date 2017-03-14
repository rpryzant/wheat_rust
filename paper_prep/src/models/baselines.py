from sklearn import svm
from sklearn.ensemble import RandomForestClassifier 
from sklearn import linear_model
from src.training.evaluation import accuracy
import numpy as np
import random


class SVM():

    def __init__(self, config):
        if config.l2 > 0:
            self.model = svm.LinearSVC(penalty='l2', C=config.l2, probability=True)
        elif config.l1 > 0:
            self.model = svm.LinearSVC(penalty='l1', C=config.l1, probability=True)
        else:
            self.model = svm.SVC(probability=True) # still regularizes

    def fit_and_predict(self, data, val_data, sess):
        x_train, y_train, _ = zip(*data)
        x_train = [np.reshape(x, [-1]).tolist() for x in x_train] # concatenate errythang
        self.model.fit(x_train, y_train)

        x_val, y_val, _ = zip(*val_data)
        x_val = [np.reshape(x, [-1]).tolist() for x in x_val] # concatenate errythang
        y_hat = self.model.predict(x_val)
        y_probs = [x[1] for x in self.model.predict_proba(x_val)]
        return y_probs, y_hat, 1


class RandomForest():
    def __init__(self, config):
        self.num_estimators = config.num_estimators
        self.max_depth = config.num_estimators
        self.model = RandomForestClassifier(n_estimators=self.num_estimators,
                                            max_depth=self.max_depth)

    def fit_and_predict(self, data, val_data, sess):
        x_train, y_train, _ = zip(*data)
        x_train = [np.reshape(x, [-1]).tolist() for x in x_train] # concatenate errythang
        self.model.fit(x_train, y_train)

        x_val, y_val, _ = zip(*val_data)
        x_val = [np.reshape(x, [-1]).tolist() for x in x_val] # concatenate errythang
        y_hat = self.model.predict(x_val)
        y_probs = [x[1] for x in self.model.predict_proba(x_val)]
        return y_probs, y_hat, 1


class LogisticRegression():
    def __init__(self, config):
        if config.l2 > 0:
            self.penalty = 'l2'
            self.c = config.l2
        elif config.l1 > 0:
            self.penalty = 'l1'
            self.c = config.l1
        else:
            self.penalty = 'l2'
            self.c = 100

        self.model = linear_model.LogisticRegression(penalty=self.penalty,
                                                     C=self.c)

    def fit_and_predict(self, data, val_data, sess):
        x_train, y_train, _ = zip(*data)
        x_train = [np.reshape(x, [-1]).tolist() for x in x_train] # concatenate errythang

        self.model.fit(x_train, y_train)
        x_val, y_val, _ = zip(*val_data)
        x_val = [np.reshape(x, [-1]).tolist() for x in x_val] # concatenate errythang
        y_hat = self.model.predict(x_val)      # TODO not very clean, do it yourself on probs
        y_probs = [x[0] for x in self.model.predict_proba(x_val)]
        return y_probs, y_hat, 1



class Random():
    # FIT A BINOMIAL
    def __init__(self, config):
        pass

    def fit_and_predict(self, data, val_data, sess):
        x_train, y_train, _ = zip(*data)
        prop_positive = np.count_nonzero(y_train) * 1.0 / len(y_train)

        x_val, y_val, _ = zip(*val_data)
        val_n = len(y_val)

        y_probs = [random.random() for _ in range(val_n)]
        y_hat = [1 if x < prop_positive else 0 for x in y_probs]

        return y_probs, y_hat, 1


