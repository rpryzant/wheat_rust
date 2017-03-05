"""
python main.py [logging loc] [completed loc]

"""


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
from src.utils.logger import Logger
import sys
import json
import os

MODEL_CLASS_MAPPINGS = {
    'lstm': LSTM,
    'conv_lstm': LSTM,
    'svm': SVM,
    'forest': RandomForest,
    'regression': LogisticRegression
}
LOGGER = Logger(sys.argv[1])
COMPLETED = Logger(sys.argv[2])


class Config():
    def __init__(self, settings={}, serialized=None):
        if serialized is not None:
            self.settings = self.deserialize(serialized)
        else:
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
        self.conv_type = settings.get('conv_type', 'max_row')


        self.l2 = settings.get('l2', 0.0)
        self.l1 = settings.get('l1', 0.0)

        self.num_estimators = settings.get('forest_estimators', 10)
        self.forest_depth = settings.get('forest_depth', None)


        self.data_path = 'datasets/score_binary_threshold_1_buckets_%s.npz' % str(settings.get('W', 32))
        self.dataset = settings.get('datset', 'standard')



    def __str__(self):
        return '\n'.join('%s \t\t %s' % (k, v) for (k, v) in self.settings.iteritems())

    def serialize(self):
        return '|'.join('%s-%s' % (k, v) for (k, v) in self.settings.iteritems())

    def deserialize(self, s):
        def isint(s):
            return s.isdigit()
        def isfloat(s):
            try:
                float(s)
                return True
            except ValueError:
                return False            

        out = {}
        for x in s.split('|'):
            [k, v] = x.split('-')
            out[k] = int(v) if isint(v) else float(v) if isfloat(v) else v
        return out



def performance(yhat, yprobs, y):
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat)
    precision = precision_score(y, yhat)
    recall = recall_score(y, yhat)
    rocs = roc_curve(y, yprobs, pos_label=1)
    return acc, f1, precision, recall, rocs





def evaluate(combo, LOGGER, COMPLETED):
    c = Config(combo)

    LOGGER.log('EVALUATING ' + c.serialize())

    start = time.time()
    LOGGER.log('\t building dataset...')
    dataset = Dataset(c.data_path)
    data_iterator = DataIterator(dataset)

    LOGGER.log('\t cross-validating...')
    preds = []
    probs = []
    val_labels = []
    model = c.model_class(c)
    for i, (val, train) in enumerate(data_iterator.xval_split(12)):
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Or whichever device you would like to use
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            val_probs, val_preds, epochs = model.fit_and_predict(train, val, sess)
            preds += list(val_preds)
            probs += list(val_probs)
            val_labels += list(zip(*val)[1])

    tf.reset_default_graph()
    acc, f1, precision, recall, rocs = performance(preds, probs, val_labels)
    rocs = map(lambda x: list(x), rocs)
    summary = {
        'acc': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'rocs': str(rocs),    # TODO  JSON DUMPS DOESNT LIKE FLOATS?????????
        'time': time.time() - start,
    }
    output = {
        'combo': c.serialize(),
        'result': summary
    }

    LOGGER.log('\t Done! Summary:')
    LOGGER.log('\t\t time: ' + str(time.time() - start))
    LOGGER.log('\t\t acc: ' + str(acc))
    LOGGER.log('\t\t f1: ' + str(f1))

    COMPLETED.log(json.dumps(output), show_time=False)

s = 'lstm_h-64|B-4|dense-256|keep_prob-0.4|L-1|dataset-standard|conv_type-max_row|lstm_conv_filters-128|model_type-conv_lstm'

#c = Config(serialized=s)
#evaluate(c, LOGGER, COMPLETED)
#evaluate(Config({'model_type': 'regression'}), LOGGER, COMPLETED)
#evaluate(Config({'model_type': 'regression'}), LOGGER, COMPLETED)


# quit()
#    sess.close()
# evaluate({})
# evaluate({'model_type': 'regression'})
# evaluate({'model_type': 'svm'})
# evaluate({'model_type': 'forest'})



def generate_configurations():
    batch_sizes = [2, 4, 8, 12]
    keep_probs = [0.25, 0.4, 0.5, 0.65]
    model_types = ['lstm', 'conv_lstm', 'svm', 'forest', 'regression']
    lstm_layers = [1, 2, 3, 4]
    lstm_hidden_size = [64, 128, 256]
    lstm_dense_size = [64, 128, 256]

    lstm_conv_filters = [16, 32, 64, 128]   
    conv_types = ['max_row', 'middle_row', 'col_pool']

    hist_buckets = [16, 25, 32, 40]

    traditional_models = []
    for mt in model_types[2:]:
        traditional_models.append({'model_type': mt})

    alt_models = []
    for bs in batch_sizes:
        for kp in keep_probs:
            for ll in lstm_layers:
                for lhs in lstm_hidden_size:
                    for lds in lstm_dense_size:
                        for nb in hist_buckets:
                            for mt in model_types[:2]:
                                if mt == 'conv_lstm':
                                    for ct in conv_types:
                                        if ct != 'max_row': continue
                                        for lcf in lstm_conv_filters:
                                            setting = {
                                                'W': nb,
                                                'B': bs,
                                                'keep_prob': kp,
                                                'L': ll,
                                                'lstm_h': lhs,
                                                'dense': lds,
                                                'model_type': mt,
                                                'dataset': 'standard',     # TODO OTHER DATASET
                                                'conv_type': 'max_row',    # TODO OTHER TYPES
                                                'lstm_conv_filters': lcf

                                            }       
                                            alt_models.append(setting)
                                else:
                                    setting = {
                                        'W': nb,
                                        'B': bs,
                                        'keep_prob': kp,
                                        'L': ll,
                                        'lstm_h': lhs,
                                        'dense': lds,
                                        'model_type': mt,
                                        'dataset': 'standard'
                                    }       
                                    alt_models.append(setting)


    random.shuffle(alt_models)
    return alt_models, traditional_models

alt, trad = generate_configurations()

#print len(alt)

for combo in alt:
    evaluate(combo, LOGGER, COMPLETED)

#Parallel(n_jobs=2)(delayed(evaluate)(combo, LOGGER, COMPLETED) for combo in alt)

