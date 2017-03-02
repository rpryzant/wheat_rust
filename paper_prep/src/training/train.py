"""

python train.py ../../data/training/score_binary-histogram_data_threshold_1.npz classification

TODO
    - EXPERIMENT WITH LABELINGS
    - initialize with crop weights
    - cut down timeseries, get more observations in there
"""
#from cnn_model import *
from lstm_model import *
import logging
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from sklearn.metrics import f1_score



def accuracy(pred, y):
    pred = [1 if x > 0.5 else 0 for x in pred]
    print pred
    print y
    return sum(1 if x == y else 0 for (x, y) in zip(pred, y)) * 1.0 / len(y)


if __name__ == "__main__":
    print 'INITIALIZING: configuration, data...'
    task_type = sys.argv[2]

    # Create a coordinator
    config = Config()

    # load data to memory
    filename = sys.argv[1]
    content = np.load(filename)

    # load and processes data
    n_timeseries = 35
    images = content['examples']
    labels = content['labels']
    filtered_indices = np.array([i for i, x in enumerate(images) if len(images[i]) == n_timeseries])  # because ragged arrays :/
    images = np.array([i for i in images[filtered_indices]])
    labels = np.array(labels[filtered_indices])

    N = len(images)
#    images = images[:config.B]   # for overfitting
#    labels = labels[:config.B]




    # load images, then
    #   -- only take images with complete timeseries info
    #   -- subtract off mean per-band histogram
    #   -- divide by sd per feature per per-band histogram
    #   -- transpose each image to get it in shape (buckets, photos, bands)
    #          (that's what the model expects)
    dim = images.shape
    concat = np.reshape(images, (-1, dim[2], dim[3]))   # concatenate images for each timeseries
    means = np.mean(concat, axis=0)
    stds = np.std(concat, axis=0)
    for i in range(len(images)):
        images[i] = (images[i] - means) / (stds + 1e-6)
    images = np.transpose(images, (0, 3, 1, 2))   

#    images = images[:, :, :, [9]]
    print images.shape
    
    train_images = images[:(N-(N/12))]
    train_labels = labels[:(N-(N/12))]
    val_images = images[(N-(N/12)):]
    val_labels = labels[(N-(N/12)):]


    print 'DATA DONE. TRAINING SET SIZE: %s TEST SET: %s' % (len(train_images), len(val_images))

    print 'INITIALIZING MODEL'
    model= NeuralModel(config,'net', task_type)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    try:
        saver.restore(sess, config.save_path)
        print 'model restored from %s' % cofig.save_path
    except:
        print 'no model found'
    print 'MODEL DONE.'

    prob_1 = np.count_nonzero(val_labels) * 1.0 / len(val_labels)
    mu_acc = 0
    f1 = 0 
    for _ in range(100):
        random_guesses = np.array([1 if random.random() < prob_1 else 0 for _ in range(len(val_labels))])
        mu_acc += accuracy(random_guesses, val_labels)
        f1 += f1_score(1 - val_labels, 1 - random_guesses)
    print 'RANDOM BASELINE ACC: ', mu_acc / 100, ' F1: ', f1 / 100

    print 'TRAINING...' 
    lr = config.lr   
    try:
        train_losses, val_losses = [], []

        for epoch in range(2000):
            if epoch % 300 == 0:
                lr *= 0.95
            epoch_train_loss = 0
            for i in range(len(train_images) / config.B):
                x_batch = train_images[i:i+config.B]
                y_batch = train_labels[i:i+config.B]

                _, train_loss, pred = sess.run([model.train_op, model.loss_err, model.y_final], feed_dict={
                    model.x: x_batch,
                    model.y: y_batch,
                    model.lr: config.lr,
                    model.keep_prob: config.drop_out
                    })
                epoch_train_loss += train_loss
            epoch_train_loss = epoch_train_loss * 1.0 / (len(train_images) / config.B)
            train_losses.append(epoch_train_loss)

            epoch_val_loss = 0
            epoch_val_acc = 0
            epoch_val_f1 = 0
            for i in range(len(val_images) / config.B):
                x_batch = val_images[i:i+config.B]
                y_batch = val_labels[i:i+config.B]

                val_loss, pred = sess.run([model.loss_err, model.y_final], feed_dict={
                    model.x: x_batch,
                    model.y: y_batch,
                    model.lr: config.lr,
                    model.keep_prob: 1.0
                    })
                epoch_val_loss += val_loss
                epoch_val_acc += accuracy(pred, y_batch)
                epoch_val_f1 += f1_score(1 - y_batch, 1 - np.array([1 if x > 0.5 else 0 for x in pred]))
                
            epoch_val_loss = epoch_val_loss * 1.0 / (len(val_images) / config.B)
            epoch_val_acc = epoch_val_acc * 1.0 / (len(val_images) / config.B)
            epoch_val_f1 = epoch_val_f1 * 1.0 / (len(val_images) / config.B)
            val_losses.append(epoch_val_loss)


            print 'epoch', epoch, 'train loss', epoch_train_loss, 'val loss', epoch_val_loss, 'acc: ', epoch_val_acc, 'f1: ', epoch_val_f1

    except KeyboardInterrupt:
        print 'stopped'

    finally:
        print 'plotting train_losses...'
        plt.plot(range(len(train_losses)), train_losses)
        plt.plot(range(len(val_losses)), val_losses)
        plt.xlabel('Epochs')
        plt.ylabel('total loss')
        plt.legend(['Train', 'Validate'])
        plt.title('loss')
        plt.savefig('LOSSES.png')
        plt.close()













