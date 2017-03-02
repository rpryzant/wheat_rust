from src.models.lstm import LSTM
from src.data.data_utils import Dataset, DataIterator
from src.training.evaluation import accuracy, Evaluator
import tensorflow as tf
from joblib import Parallel, delayed
import random
import time
from src.models.sklearn_models import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve
import random



MODEL_CLASS_MAPPINGS = {
    'lstm': LSTM,
    'conv_lstm': LSTM,
    'svm': SVM,
    'forest': RandomForest,
    'regression': LogisticRegression
}


class Config():
    def __init__(self, settings={}):
        self.settings = settings

        self.B = settings.get('B', 8)
        self.W = settings.get('W', 32)
        self.H = settings.get('H', 35)
        self.C = settings.get('C', 10)

        self.layers = settings.get('L', 2)
        self.lstm_h = settings.get('lstm_h', 128)
        self.dense = settings.get('dense', 128)
        self.lr = settings.get('lr', 0.0003)
        self.keep_prob = settings.get('keep_prob', 0.50)

        self.model_type = settings.get('model_type', 'lstm')
        self.model_class = MODEL_CLASS_MAPPINGS[self.model_type]

        self.num_lstm_filters = settings.get('lstm_conv_filters', 128)

        self.l2 = settings.get('l2', 0.0)
        self.l1 = settings.get('l1', 0.0)

        self.num_estimators = settings.get('forest_estimators', 10)
        self.forest_depth = settings.get('forest_depth', None)


        self.data_path = 'datasets/score_binary-histogram_data_1.npz'


    def __str__(self):
        return '\n'.join('%s \t %s' % (k, v) for (k, v) in self.settings.iteritems())


def performance(yhat, yprobs, y):
    print yhat
    print yprobs
    print y
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat)
    precision = precision_score(y, yhat)
    recall = recall_score(y, yhat)
    rocs = roc_curve(y, yprobs, pos_label=1)
    return acc, f1, precision, recall, rocs





def evaluate(combo):

    c = Config(combo)
    print '======================='
    print 'EVALUATING'
    print str(combo)

    start = time.time()
    print '\t building dataset...'
    dataset = Dataset(c.data_path)
    data_iterator = DataIterator(dataset)

    print '\t cross-validating...'
    preds = []
    probs = []
    val_labels = []
    model = c.model_class(c)
    for i, (val, train) in enumerate(data_iterator.xval_split(12)):
        print '\t split %s...' % i
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            val_probs, val_preds, epochs = model.fit_and_predict(train, val, sess)
            preds += list(val_preds)
            probs += list(val_probs)
            val_labels += list(zip(*val)[1])

    tf.reset_default_graph()

    acc, f1, precision, recall, rocs = performance(preds, probs, val_labels)
    print '======================================'
    print str(c)
    print 'COMBO:            ', combo
    print 'TOOK              ', time.time() - start, ' seconds'
    print 'ACC               ', acc 
    print 'F1                ', f1
    print 'PRE               ', precision
    print 'REC               ', recall
    print '======================================='




#    sess.close()
evaluate({'model_type': 'regression'})
evaluate({})
evaluate({'model_type': 'svm'})
evaluate({'model_type': 'forest'})



def generate_configurations():
    batch_sizes = [2, 4, 8, 12]
    keep_probs = [0.25, 0.4, 0.5, 0.65]
    model_types = ['lstm', 'conv_lsvm', 'svm', 'forest', 'regression']
    lsm_layers = [1, 2, 3]
    lstm_hidden_size = [64, 128, 256]
    lstm_dense_size = [64, 128, 256]
    lstm_conv_filters = [16, 32, 64, 128]   

    # while True:
    #     config = {
    #         'B':
    #     }


#for combo in combos:
#    evaluate(combo)

Parallel(n_jobs=2)(delayed(evaluate)(combo) for combo in combos)

