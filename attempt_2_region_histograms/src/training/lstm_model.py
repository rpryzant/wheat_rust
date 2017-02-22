import numpy as numpy
import tensorflow as tf
import sys



class Config():
    B, W, H, C = 5, 32, 35, 10

    layers = 3

    lstm_h = 128

    dense = 256

    train_step = 10000
#    lr = 1.0
    lr = 0.0003
    drop_out = 0.75



def run_affine(inputs, H, N=None, name="affine_layer"):
    if not N:
        N = inputs.get_shape()[-1]
    with tf.variable_scope(name):
        print N, H
        W = tf.get_variable("W", [N, H], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [1, H])
        return tf.matmul(inputs, W) + b


def run_lstm(inputs, targets, config, keep_prob=1):
    cell = tf.nn.rnn_cell.LSTMCell(config.lstm_h, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell([cell] * config.layers, state_is_tuple=True)
    state = cell.zero_state(config.B, tf.float32)
    print state
    outputs, final_state = tf.nn.dynamic_rnn(cell,
                                             inputs,
                                             initial_state=state,
                                             time_major=True)
    final_outputs = outputs[-1]
    return final_outputs


#shape=(35, 32, 128)

class NeuralModel():
    def __init__(self, config, name, task):
        self.x = tf.placeholder(tf.float32, [None, config.W, config.H, config.C], name='x')
        self.y = tf.placeholder(tf.float32, [None])
        self.lr = tf.placeholder(tf.float32, [])
        self.keep_prob = tf.placeholder(tf.float32, [])

        inputs = tf.transpose(self.x, [2, 0, 1, 3])   # move time to first dimension
        dim = inputs.get_shape().as_list()
        inputs = tf.reshape(inputs, [dim[0], -1, dim[2]*dim[3]])  # concat bands for each image

        lstm_out = run_lstm(inputs, self.y, config, keep_prob=self.keep_prob)
        fc1 = run_affine(lstm_out, config.dense, name='fc1')
        self.logits = tf.squeeze(run_affine(fc1, 1, name='logits'))
        if task == "classification":
            self.y_final = tf.sigmoid(self.logits)
            self.loss_err = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.y))
        else:
            self.y_final = self.logits
            self.loss_err = tf.nn.l2_loss(self.logits - self.y)
#        self.train_op = tf.contrib.layers.optimize_loss(self.loss_err, None, self.lr, 'SGD', clip_gradients=5.)

        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_err)




