"""


"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
import random
import time

class LSTM(object):
    
    def __init__(self, num_classes, num_features, max_seq_len, learning_rate, hidden_units, model_path=None):
        self.num_classes = num_classes
        self.num_features = num_features
        self.max_seq_len = max_seq_len
        self.lr = learning_rate
        self.hidden_units = hidden_units

        # placeholders for data
        self.input = tf.placeholder(tf.float32, shape=[None, max_seq_len, num_features])
        self.target = tf.placeholder(tf.int32, shape=[None, num_classes])
        self.length = tf.placeholder(tf.int32, shape=[None])       # expect variable length inputs
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')  # for counting interations

        # forward and backward passes
        logits = self.__forward_pass()
        self.pred = tf.nn.softmax(logits)
        self.loss = self.__batch_loss(logits)       # mean loss across a batch 
        self.train_step = self.__backward_pass()

        # tf session boilerplate
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        # restore saved params
        self.saver = tf.train.Saver()
        if model_path is not None:
            self.saver.restore(self.sess, model_path)


    def __backward_pass(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        return train_step


    def __batch_loss(self, logits):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits, self.target)
        loss_per_batch = tf.reduce_sum(losses) / tf.to_float(self.length)
        mean_batch_loss = tf.reduce_mean(loss_per_batch)
        return mean_batch_loss


    def __forward_pass(self):
        # declare and unravel lstm cell
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_units)
        outputs, state = tf.nn.dynamic_rnn(
            cell=cell,
            inputs=self.input,
            sequence_length=self.length,
            dtype=tf.float32)

        # pull out outputs[-1] (TF doesn't support negative indexing)
        outputs = tf.transpose(outputs, [1, 0, 2])
        last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

        # send through fc layer to get logits
        V = tf.get_variable(
            name='V',
            initializer=tf.contrib.layers.xavier_initializer(),
            shape=[self.hidden_units, self.num_classes])
        logits = tf.matmul(last_output, V)

        return logits


    def __pad(self, x):
        """pads a single example and returns its original length
        """
        l = len(x)
        while len(x) < self.max_seq_len:
            x.append(np.zeros_like(x[0]))
        return x, l


    def __prep_x(self, x):
        [x_padded, lengths] = zip(*[self.__pad(xi) for xi in x])        
        return list(x_padded), list(lengths)


    def save_params(self, path):
        """ save current graph into a tf checkpoint file
            "path" should be a string that includes a directory and the prefix
            for your checkpoints (these checkpoints will have a "-[iter #]" suffix)
        """
        self.saver.save(self.sess, path, global_step=self.global_step)


    def train_on_batch(self, x_batch, y_batch):
        x_batch, l_batch = self.__prep_x(x_batch)
        _, loss = self.sess.run([self.train_step, self.loss],
                           feed_dict={
                               self.input: x_batch,
                               self.target: y_batch,
                               self.length: l_batch
                           })
        return loss


    def predict(self, X):
        X, L = self.__prep_x(X)
        pred_dist = self.sess.run(self.pred, 
                                   feed_dict={
                                       self.input: X,
                                       self.length: L
                                   })
        return np.argmax(pred_dist, axis=1)





if __name__ == "__main__":
    MAX_SEQ_LEN = 200
    BATCH_SIZE = 3
    VOCAB_SIZE = 8001 # 1 extra for padding

    SEQ_LEN = 10
    NUM_FEATURES = 4
    NUM_EXAMPLES = 1000
    BATCH_SIZE = 16
    NUM_CLASSES = 2      # diseased / healthy (leaving door open for more classes later)


    def accuracy(Y1, Y2):
        return sum(1 if x == y else 0 for (x, y) in zip(Y1, Y2)) * 1.0 / len(Y1)

    print '=== TESTING...' 
    
    # make some fake data
    X = [ [np.random.normal(size=(NUM_FEATURES)) for _ in range(random.randint(1, SEQ_LEN))] for _ in range(NUM_EXAMPLES) ]
    Y = [[1, 0] if random.random() < 0.5 else [0, 1] for _ in range(NUM_EXAMPLES)]

    print '=== BUILDING MODEL'
    start = time.time()
    model = LSTM(num_classes=NUM_CLASSES, 
                max_seq_len=SEQ_LEN, 
                num_features=NUM_FEATURES,
                learning_rate=0.0003, 
                hidden_units=128)
    print '=== TOOK {:.2f} SECONDS\n'.format(time.time() - start)


    y_pred = model.predict(X)
    y_true =  np.array([y for [_ , y] in  Y])
    initial_acc = accuracy(y_pred, y_true)

    print '=== TRAINING'
    start = time.time()
    for epoch in tqdm(range(10)):
        epoch_loss = 0
        for i in range(0, len(X) - BATCH_SIZE)[::BATCH_SIZE]:
            y_batch = Y[i:i+BATCH_SIZE]
            x_batch = X[i:i+BATCH_SIZE]
            epoch_loss += model.train_on_batch(x_batch, y_batch)
#        print epoch_loss
    print '=== TOOK {:.2f} SECONDS\n'.format(time.time() - start)

    y_pred = model.predict(X)
    acc = accuracy(y_pred, y_true)

    if initial_acc < acc:
        print '=== TEST PASSES: ACCURACY IMPROVED'

















