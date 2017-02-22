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
            print 'using sigmoid xentropy loss'
            self.y_final = tf.sigmoid(self.logits)
            self.loss_err = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.logits, self.y))
        else:
            print 'using l2 loss'
            self.y_final = self.logits
            self.loss_err = tf.nn.l2_loss(self.logits - self.y)

        self.loss_reg = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

        self.loss = self.loss_err #+ (0.2 * self.loss_reg)

#        self.train_op = tf.contrib.layers.optimize_loss(self.loss_err, None, self.lr, 'SGD', clip_gradients=5.)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)



if __name__ == '__main__':
    test_type = sys.argv[1]

    sess = tf.Session()
    config = Config()
    model = NeuralModel(config, "model", test_type)
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
            model.keep_prob: config.drop_out
        })

        losses.append(loss)

    print 'INFO: plotting losses...'
    plt.plot(range(len(losses)), losses)
    plt.xlabel('Epochs')
    plt.ylabel('total loss')
    plt.title('loss')
    plt.savefig('LOSSES_test.png')
    plt.close()





