"""
This is the main.py for all the transfer learning stuff

"""
import numpy as np
from src.models.lstm import LSTM
from main import Config
import tensorflow as tf
import os

#print 'HERE'
#quit()
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



predict_year = 2013
data_path = "/atlas/u/jiaxuan/data/google_drive/img_output/"

filename = 'histogram_all' + '.npz'

content = np.load(data_path + filename)
image_all = content['output_image']
yield_all = content['output_yield']
year_all = content['output_year']
locations_all = content['output_locations']
index_all = content['output_index']


 # delete broken image
list_delete=[]
for i in range(image_all.shape[0]):
    if np.sum(image_all[i,:,:,:])<=287:
        if year_all[i]<2016:
            list_delete.append(i)
image_all=np.delete(image_all,list_delete,0)
yield_all=np.delete(yield_all,list_delete,0)
year_all = np.delete(year_all,list_delete, 0)
locations_all = np.delete(locations_all, list_delete, 0)
index_all = np.delete(index_all, list_delete, 0)


# keep major counties
list_keep=[]
for i in range(image_all.shape[0]):
    if (index_all[i,0]==5)or(index_all[i,0]==17)or(index_all[i,0]==18)or(index_all[i,0]==19)or(index_all[i,0]==20)or(index_all[i,0]==27)or(index_all[i,0]==29)or(index_all[i,0]==31)or(index_all[i,0]==38)or(index_all[i,0]==39)or(index_all[i,0]==46):
        list_keep.append(i)
image_all=image_all[list_keep,:,:,:]
yield_all=yield_all[list_keep]
year_all = year_all[list_keep]
locations_all = locations_all[list_keep,:]
index_all = index_all[list_keep,:]

# split into train and validate
index_train = np.nonzero(year_all < predict_year)[0]
index_validate = np.nonzero(year_all == predict_year)[0]
print 'train size',index_train.shape[0]
print 'validate size',index_validate.shape[0]

# calc train image mean (for each band), and then detract (broadcast)
image_mean=np.mean(image_all[index_train],(0,1,2))
image_all = image_all - image_mean

image_validate=image_all[index_validate]
yield_validate=yield_all[index_validate]



s = 'H-32|C-9|lstm_h-128|B-2|dense-64|lstm_conv_filters-64|model_type-conv_lstm|keep_prob-0.5|L-1|conv_type-valid|dataset-standard'


# s = {
#     'H': 32,
#     'model_type': 'lstm',
#     'C': 9,
#     'B': 2
# }

c = Config(s)
model = LSTM(c, regression=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # Or whichever device you would like to use
gpu_options = tf.GPUOptions(allow_growth=True)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())

    batch_size = 2
    for epoch in range(10):
        i = 0
        epoch_loss = 0
        while i + batch_size < len(image_all):
            lengths = [len(x) for x in image_all[i: i + batch_size]]
            _, loss = sess.run([model.train_op, model.loss], feed_dict={
                    model.x: image_all[i: i + batch_size],
                    model.y: yield_all[i: i + batch_size],
                    model.l: lengths,
                    model.lr: c.lr,
                    model.keep_prob: c.keep_prob
                })
            i += batch_size
            epoch_loss += loss
        print epoch_loss / i
        saver.save(sess, 'checkpoints/transfer', global_step=model.global_step)




