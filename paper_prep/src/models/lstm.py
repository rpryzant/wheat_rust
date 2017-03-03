# -*- coding: utf-8 -*-
"""
unit tests:
python lstm_model.py classification
python lstm_model.py regression
"""


import numpy as numpy
import tensorflow as tf
import sys
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.training.evaluation import accuracy


class Config():
    B, W, H, C = 5, 32, 35, 10 #1
#    B, W, H, C = 5, 32, 25, 10 #1

    layers = 3

    lstm_h = 256

    dense = 256

    train_step = 10000
#    lr = 1.0
    lr = 0.0003
    drop_out = 0.75
    save_path = '../../data/jiaxuans_models/hist_hidden_256/2015CNN_model'




def conv_relu_batch(input_data, filter_dims, stride, name="crb"):
    def conv2d(name="conv2d"):
        with tf.variable_scope(name):
            W = tf.get_variable("W", filter_dims,
                    initializer=tf.contrib.layers.variance_scaling_initializer())
            b = tf.get_variable("b", [1, 1, 1, filter_dims[-1]])
            return tf.nn.conv2d(input_data, W, [1, stride, stride, 1], "SAME") + b

    def batch_normalization(input_data, axes=[0], name="batch"):
        with tf.variable_scope(name):
            mean, variance = tf.nn.moments(input_data, axes, keep_dims=True, name="moments")
            return tf.nn.batch_normalization(input_data, mean, variance, None, None, 1e-6, name="batch")

    with tf.variable_scope(name):
        a = conv2d()
        b = batch_normalization(a,axes=[0,1,2])
        r = tf.nn.relu(b)
        return tf.reduce_max(r, axis=2)    # give the maximal activation over each row


def run_affine(inputs, H, N=None, name="affine_layer"):
    if not N:
        N = inputs.get_shape()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [1, H], initializer=tf.constant_initializer(0))
        return tf.matmul(inputs, W) + b


def run_lstm(inputs, targets, config, keep_prob=1):
    cell = tf.nn.rnn_cell.LSTMCell(config.lstm_h, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.layers, state_is_tuple=True)
    state = cell.zero_state(config.B, tf.float32)
    outputs, final_state = tf.nn.dynamic_rnn(cell,
                                             inputs,
                                             initial_state=state,
                                             time_major=True)
    final_outputs = outputs[-1]
    return final_outputs


#shape=(35, 32, 128)

class LSTM():
    def __init__(self, config):
#        self.sess = sess
        self.config = config
#        with tf.variable_scope(name):
        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name='x')
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        self.batch_size = config.B


        if self.config.model_type == 'conv_lstm':
            # [batch, in_height, in_width, in_channels]
            inputs = tf.transpose(self.x, [0, 2, 1, 3])

            # [filter_height, filter_width, in_channels, out_channels]
            filter_dims = [1, self.config.W, config.C, config.num_lstm_filters]

            lstm_inputs = conv_relu_batch(inputs, filter_dims, 1)
            lstm_inputs = tf.transpose(lstm_inputs, [1, 0, 2])  # time to 1st dim
        else:
            inputs = tf.transpose(self.x, [2, 0, 1, 3])   # move time to first dimension
            dim = inputs.get_shape().as_list()
            inputs = tf.reshape(inputs, [dim[0], -1, dim[2]*dim[3]])  # concat bands for each image
            lstm_inputs = inputs

        lstm_out = run_lstm(lstm_inputs, self.y, config, keep_prob=self.keep_prob)
        fc1 = run_affine(lstm_out, config.dense, name='fc1')
        self.logits = tf.squeeze(run_affine(fc1, 1, name='logits'))
        self.y_final = tf.sigmoid(self.logits)

        self.loss_err = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.y))

        self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        self.loss = self.loss_err #+ (config.l2 * self.loss_reg)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



    def fit_and_predict(self, data, val_data, sess):

        def train_epoch(data):
            i = 0
            train_loss = 0
            while i + self.batch_size < len(data):
                batch = data[i: i + self.batch_size]
                x_batch, y_batch = zip(*batch)
                _, loss = sess.run([self.train_op, self.loss], feed_dict={
                    self.x: x_batch,
                    self.y: y_batch,
                    self.lr: self.config.lr,
                    self.keep_prob: self.config.keep_prob
                })
                train_loss += loss
                i += self.batch_size
            train_loss /= (i / self.batch_size)
            return train_loss


        def predict(data):
            i = 0
            out_pred = []
            out_prob = []
            total_loss = 0.0
            while i + self.batch_size < len(data):
                batch = data[i: i + self.batch_size]
                x_batch, y_batch = zip(*batch)
                loss, pred = sess.run([self.loss, self.y_final], feed_dict={
                    self.x: x_batch,
                    self.y: y_batch,
                    self.keep_prob: 1.0
                    })
                total_loss += loss
                out_prob += [x for x in pred]
                out_pred = out_pred + [1 if x > 0.5 else 0 for x in pred]
                i += self.batch_size

            # batches dont fit into data nicely, so get the last batch
            # this is SO hacky and gross and i'm completely disgusted with 
            # myself but here we are...  ¯\_(ツ)_/¯ (oh and im also ignoring
            # the loss from this little overhang. sigh.)
            # whoever's reading this....i'm so sorry will you ever find 
            # room in your heart for forgiveness\
            # !!!!!!! TODO -- REFACTOR !!!!!!!!!
            final_batch = data[-self.batch_size:]
            x_batch, y_batch = zip(*final_batch)
            loss, pred = sess.run([self.loss, self.y_final], feed_dict={
                self.x: x_batch,
                self.y: y_batch,
                self.keep_prob: 1.0
                })
            remainder = len(data) - i
            to_add = pred[-remainder:]
            out_prob = out_prob + list(to_add[:])
            out_pred = out_pred + [1 if x > 0.5 else 0 for x in to_add[:]]
            acc = accuracy(out_pred, zip(*data)[1])

            return out_prob, out_pred, total_loss / (i / self.batch_size), acc

        epoch = 0
        best_loss = float('inf')
        best_acc = -float('inf')

        prob, pred, loss, acc = predict(val_data)
        best_preds = pred, prob

        train_epoch(data)

        epochs = 1
        while loss < best_loss or acc > best_acc:
            best_loss = loss if loss < best_loss else best_loss
            if acc > best_acc:
                best_acc = acc
                best_preds = (pred, prob)

            prob, pred, loss, acc = predict(val_data)
#            print loss, acc, epochs
            train_epoch(data)
            epochs += 1
        best_pred, best_prob = best_preds      #sigh
        return best_prob, best_pred, epochs



if __name__ == '__main__':
    test_type = sys.argv[1]

    sess = tf.Session()
    config = Config()
    model = LSTM(config, "model", test_type)
    sess.run(tf.initialize_all_variables())

    print 'INFO: running %s test' % test_type
    dummy_x = np.random.rand(config.B, config.W, config.H, config.C)
    dummy_y = np.random.randint(2, size=config.B) if test_type == 'classification' else np.random.rand(config.B)
    print dummy_x.shape
    losses = []
    for i in tqdm(range(1000)):
        # model.state = model.cell.zero_state(config.B, tf.float32)
        if i % 100 == 0:
            config.lr /= 2
        _, loss, pred = sess.run([model.train_op, model.loss, model.y_final], feed_dict={
            model.x: dummy_x,
            model.y: dummy_y,
            model.lr: config.lr,
            model.keep_prob: config.keep_prob
        })

        losses.append(loss)

    print 'INFO: plotting losses...'
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('total loss')
    plt.title('loss')
    plt.savefig('LOSSES_test.png')
    plt.close()





