"""
python main.py [logging loc] [completed loc]

"""


from src.models.cnn import CNN
from src.models.lstm import LSTM
from src.data.data_utils import Dataset, DataIterator
from src.training.evaluation import accuracy, Evaluator
import tensorflow as tf
#from joblib import Parallel, delayed   # because atlas 5 doesn't have this
import random
import time
from src.models.sklearn_models import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, roc_auc_score
import random
from src.utils.logger import Logger
import sys
import json
import os



class Config():
    def __init__(self, settings={}):
        MODEL_CLASS_MAPPINGS = {
            'lstm': LSTM,
            'conv_lstm': LSTM,
            'conv': CNN,
            'svm': SVM,
            'forest': RandomForest,
            'regression': LogisticRegression
        }

        if type(settings) == type('string'):
            self.settings = deserialize(serialized)
        else:
            self.settings = settings

        self.B = settings.get('B', 8)
        self.W = settings.get('W', 32)
        self.H = settings.get('H', 35)
        self.C = settings.get('C', 10)

        self.deletion_band = settings.get('deletion_band', -1)

        self.layers = settings.get('L', 2)
        self.lstm_h = settings.get('lstm_h', 128)
        self.dense = settings.get('dense', 128)
        self.lr = settings.get('lr', 0.0003)
        self.keep_prob = settings.get('keep_prob', 0.50)

        self.model_type = settings.get('model_type', 'lstm')
        self.model_class = MODEL_CLASS_MAPPINGS[self.model_type]

        self.num_lstm_filters = settings.get('lstm_conv_filters', 128)
        self.conv_type = settings.get('conv_type', 'valid')

        self.l2 = settings.get('l2', 0.0)
        self.l1 = settings.get('l1', 0.0)

        # size of conv filter banks
        self.l1_n = settings.get('l1_n', 128)
        self.l2_n = settings.get('l2_n', 128)
        self.l3_n = settings.get('l3_n', 128)
        # filter size
        self.filter_size = settings.get('filter_size', 3)

        self.num_estimators = settings.get('forest_estimators', 10)
        self.forest_depth = settings.get('forest_depth', None)


        self.data_path = 'datasets/score_binary_threshold_1_buckets_%s.npz' % str(settings.get('W', 32))
        self.dataset = settings.get('datset', 'standard')



    def __str__(self):
        return '\n'.join('%s \t\t %s' % (k, v) for (k, v) in self.settings.iteritems())

    def serialize(self):
        return '|'.join('%s-%s' % (k, v) for (k, v) in self.settings.iteritems())

def deserialize(s):
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
    yhat = np.array(yhat)
    yprobs = np.array(yprobs)
    y = np.array(y)
    acc = accuracy_score(y, yhat)
    f1 = f1_score(y, yhat)
    precision = precision_score(1 - y, 1 - yhat)
    recall = recall_score(1 - y, 1 - yhat)
    rocs = roc_curve(y, yprobs, pos_label=1)
    auc = roc_auc_score(y, yprobs)
    return acc, f1, precision, recall, rocs, auc





def evaluate(combo, LOGGER, COMPLETED):
    c = Config(combo)

    LOGGER.log('EVALUATING ' + c.serialize())

    start = time.time()
    LOGGER.log('\t building dataset...')
    dataset = Dataset(c.data_path, c)
    data_iterator = DataIterator(dataset)
    quit()
    LOGGER.log('\t cross-validating...')
    preds = []
    probs = []
    val_labels = []
    model = c.model_class(c)
    for i, (val, train) in enumerate(data_iterator.xval_split(12)):
        print '\t\t split ', i
        os.environ['CUDA_VISIBLE_DEVICES'] = '2' # Or whichever device you would like to use
        gpu_options = tf.GPUOptions(allow_growth=True)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            val_probs, val_preds, epochs = model.fit_and_predict(train, val, sess)
            preds += list(val_preds)
            probs += list(val_probs)
            val_labels += list(zip(*val)[1])

    tf.reset_default_graph()
    acc, f1, precision, recall, rocs, auc = performance(preds, probs, val_labels)
    rocs = map(lambda x: list(x), rocs)
    summary = {
        'acc': acc,
        'f1': f1,
        'auc': auc,
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













def generate_configurations():
    traditional_models = []
    for mt in ['regression', 'forest', 'svm']:
        traditional_models.append({'model_type': mt})

    alt_models = []
    for batch_size in [2]:
        for keep_prob in [0.5]:
            for model_type in ['lstm', 'conv_lstm', 'conv']:
                for dense_size in [64]:
                    for buckets in [30, 35, 40, 45, 50, 55, 60, 65]:
                        for lstm_h in [64, 128, 256, 512]:

                            if model_type == 'conv_lstm':
                                for conv_type in ['pool', 'valid', '2d']:
                                    for num_filters in [16, 32, 64]:
                                        s = {
                                            'W': buckets,
                                            'B': batch_size,
                                            'keep_prob': keep_prob,
                                            'L': 1,
                                            'lstm_h': lstm_h,
                                            'dense': dense_size,
                                            'model_type': model_type,
                                            'dataset': 'standard',
                                            'conv_type': conv_type,
                                            'lstm_conv_filters': num_filters,
                                        }
                                        alt_models.append(s)

                            elif model_type == 'conv':
                                for layer_1_num_filters in [128, 256, 512]:
                                    for layer_2_num_filters in [128, 256, 512]:
                                        for layer_3_num_filters in [128, 256, 512]:
                                            for filter_size in [3, 5, 9]:
                                                s = {
                                                    'l1_n': layer_1_num_filters,
                                                    'l2_n': layer_3_num_filters,
                                                    'l3_n': layer_3_num_filters,
                                                    'W': buckets,
                                                    'B': batch_size,
                                                    'keep_prob': keep_prob,
                                                    'L': 1,
                                                    'lstm_h': lstm_h,
                                                    'dense': dense_size,
                                                    'model_type': model_type,
                                                    'dataset': 'standard',
                                                    'conv_type': conv_type,
                                                    'lstm_conv_filters': num_filters,
                                                    'filter_size': filter_size                                            
                                                }
                                                alt_models.append(s)

                            else:
                                s = {
                                    'W': buckets,
                                    'B': batch_size,
                                    'keep_prob': keep_prob,
                                    'L': 4,
                                    'lstm_h': lstm_h,
                                    'dense': dense_size,
                                    'model_type': 'lstm',
                                    'dataset': 'standard'
                                }
                                alt_models.append(s)



    random.shuffle(alt_models)
    return alt_models, traditional_models

if __name__ == '__main__':

    LOGGER = Logger(sys.argv[1])
    COMPLETED = Logger(sys.argv[2])

    # del_index in [0, 9]
    s = 'lstm_h-64|B-2|dense-64|W-40|model_type-lstm|keep_prob-0.5|L-4|dataset-standard|C-9|deletion_band-0'
    #s = 'lstm_h-256|B-2|dense-64|lstm_conv_filters-64|W-40|model_type-conv_lstm|keep_prob-0.65|L-1|conv_type-max_row|dataset-standard'
    evaluate(deserialize(s), LOGGER, COMPLETED)
    quit()
    #evaluate({'model_type': 'conv', 'W': 40, 'dense':64}, LOGGER, COMPLETED)

    #quit()

    #evaluate(Config({'model_type': 'regression'}), LOGGER, COMPLETED)
    #evaluate(Config({'model_type': 'regression'}), LOGGER, COMPLETED)


    # quit()
    #    sess.close()
    # evaluate({})
    # evaluate({'model_type': 'regression'})
    # evaluate({'model_type': 'svm'})
    # evaluate({'model_type': 'forest'})



    alt, trad = generate_configurations()
    print len(alt)
    #quit()
    #print len(alt)

    for combo in alt:
        print combo
        evaluate(combo, LOGGER, COMPLETED)

    #Parallel(n_jobs=2)(delayed(evaluate)(combo, LOGGER, COMPLETED) for combo in alt)

