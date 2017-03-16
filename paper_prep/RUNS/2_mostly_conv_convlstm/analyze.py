"""
This is a quick script for finding the best configuration for some hyperparameters
"""
import json
import sys

outs = [json.loads(l.strip()) for l in open('outs')]


#print "Best conv lstm"
#  .742   lstm_h-128|B-2|dense-64|lstm_conv_filters-64|W-40|model_type-conv_lstm|keep_prob-0.5|L-1|conv_type-valid|dataset-standard
print max((d['result']['auc'], d) for d in outs if 'model_type-conv_lstm' in d['combo'])[1]



#print "Best lstm"
#print max((d['result']['auc'], d) for d in outs if 'model_type-lstm' in d['combo'])[1]

#print 'Best conv'
# 0.7089   B-2|l3_n-256|keep_prob-0.5|L-1|dataset-standard|filter_size-9|W-40|conv_type-2d|l2_n-256|lstm_h-128|dense-64|l1_n-256|lstm_conv_filters-64|model_type-conv
#print max((d['result']['acc'], d) for d in outs if d['combo'].endswith('model_type-conv')  )[1]

