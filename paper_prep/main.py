


from src.models.lstm import LSTM
from src.models.svm import SVM
from src.data.data_utils import Dataset, DataIterator
from src.training.evaluation import accuracy, Evaluator
import tensorflow as tf
from joblib import Parallel, delayed
import random


class Config():
    def __init__(self):
        self.B, self.W, self.H, self.C = 5, 32, 35, 10
        self.layers = 3
        self.lstm_h = 256
        self.dense = 256
        self.train_step = 10000
        self.lr = 0.0003
        self.keep_prob = 0.50

        self.model_type = 'conv_lstm'
        self.num_lstm_filters = 128

        self.l2 = 0.0  # amount of l2 reg
        self.l1 = 0.0

        self.data_path = 'datasets/score_binary-histogram_data_1.npz'


    def __str__(self):
        print 'b, w, h, c: ', self.B, self.W, self.H, self.C
        print 'layers: ', self.layers
        print 'lstm h: ', self.lstm_h
        print 'dense: ', self.dense
        print 'keep_prob: ', self.keep_prob
        print 'model_type: ', self.model_type
        print 'num lstm filters: ', self.num_lstm_filters
        return ''


c = Config()

def evaluate(combo):
    c = Config()

    c.B = combo[0]
    c.layers = combo[1]
    c.lstm_h = combo[2]
    c.dense = combo[3]
    c.keep_prob = combo[4]
    c.model_type = combo[5]
    if len(combo) > 6:
        c.num_lstm_filters = combo[6]

    model = LSTM(c)

    accs = 0
    for val, train in data_iterator.xval_split(12):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss, acc, epochs = model.fit_and_predict(train, val, sess)
#            print 'FINAL: ', loss, acc, epochs
            accs += acc
    print '======================================'
    print str(config)
    print 'AVERAGE ACC       ', accs * 1.0 / 12
    print '======================================='
#    sess.close()

dataset = Dataset(c.data_path)

data_iterator = DataIterator(dataset)

#l = SVM(c)
combos = []
for b in [1, 3, 5, 7, 12]:
    for l in [1, 2, 3]:
        for h in [128, 256, 512]:
            for d in [128, 256, 512]:
                for kp in [0.25, 0.4, 0.5, 0.7]:
                    for model_type in ['lstm', 'conv_lstm']:
                        if model_type == 'conf_lstm':
                            for nf in [32, 64, 128]:
                                combos.append([b, l, h, d, kp, model_type, nf])
                        else:
                            combos.append([b, l, h, d, kp, model_type])
random.shuffle(combos)
Parallel(n_jobs=4)(delayed(evaluate)(combo) for combo in combos)

