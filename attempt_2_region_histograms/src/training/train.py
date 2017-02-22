"""

python train.py ../../data/training/score_binary-histogram_data.npz classification

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
    images = np.array([x for x in images if len(x) == n_timeseries])
    labels = content['labels']
    N = len(images)
#    images = images[:config.B]
#    labels = labels[:config.B]




    locations = content['ids']
#    indices = np.arange(len(images))
#    np.random.shuffle(indices)
#    N = len(images)

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

    train_images = images[:(N-(N/8))]
    train_labels = labels[:(N-(N/8))]
    val_images = images[(N-(N/8)):]
    val_labels = labels[(N-(N/8)):]


    print 'DATA DONE. TRAINING SET SIZE: %s TEST SET: %s' % (len(train_images), len(val_images))

    print 'INITIALIZING MODEL'
    model= NeuralModel(config,'net', task_type)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print 'MODEL DONE.'



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
            epoch_train_loss = epoch_train_loss * 1.0 / len(train_images)
            train_losses.append(epoch_train_loss)

            epoch_val_loss = 0
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
            epoch_val_loss = epoch_val_loss * 1.0 / len(val_images)
            val_losses.append(epoch_val_loss)


            print 'epoch', epoch, 'train loss', epoch_train_loss, 'val loss', epoch_val_loss
            print
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













    quit()
###############################################################################3




    RMSE_min = 100
    try:
        for i in range(config.train_step):
            if i==350:
                config.lr/=10
            if i==2000:
                config.lr/=10

            # TODO - COULD TRY AUGMENTATION THING HE DOES
            batch_indices = np.random.choice(train_indices, size=config.B)

            _, train_loss = sess.run([model.train_op, model.loss_err], feed_dict={
                    model.x: images[batch_indices],
                    model.y: labels[batch_indices],
                    model.lr: config.lr,
                    model.keep_prob: config.drop_out
                })

            if i%20 == 0:
                # do validation
                # TODO -RESTORE REAL VALIDATION
#                val_batch_indices = np.random.choice(val_indices, size=config.B) 
                val_batch_indices = np.random.choice(train_indices, size=config.B)
                val_loss = sess.run([model.loss_err], feed_dict={
                    model.x: images[val_batch_indices],
                    model.y: labels[val_batch_indices],
                    model.keep_prob: 1
                })

                pred = []
                real = []
                target_op = model.y_final if task_type == 'classification' else model.logits
                for j in range(val_images.shape[0] / config.B):
                    val_batch_indices = np.arange(j*config.B, (j+1)*config.B)
                    gold_labels = labels[val_batch_indices]
                    predictions = sess.run(target_op, feed_dict={
                        model.x: images[val_batch_indices],
                        model.y: gold_labels,
                        model.keep_prob: 1
                        })
                    # TODO -RESTORE THIS 
          #          if task_type == 'classification':        
           #             predictions = [1 if x > 0.5 else 0 for x in predictions]
                    pred.append(predictions)
                    real.append(gold_labels)
                pred=np.concatenate(pred)
                real=np.concatenate(real)


                print pred.tolist()
                print real.tolist()
                accuracy = sum(1 if x == y else 0 for (x, y) in zip(pred, real)) * 1.0 / len(pred)
                RMSE=np.sqrt(np.mean((pred-real)**2))
                ME=np.mean(pred-real)

                if RMSE<RMSE_min:
                    RMSE_min=RMSE
                    # # save
                    # save_path = saver.save(sess, config.save_path + str(predict_year)+'CNN_model.ckpt')
                    # print('save in file: %s' % save_path)
                    # np.savez(config.save_path+str(predict_year)+'result.npz',
                    #     summary_train_loss=summary_train_loss,summary_eval_loss=summary_eval_loss,
                    #     summary_RMSE=summary_RMSE,summary_ME=summary_RMSE)

                print 'Validation set','RMSE',RMSE,'ME',ME,'RMSE_min',RMSE_min, 'ACC', accuracy
                logging.info('Validation set RMSE %f ME %f RMSE_min %f ACC %f',RMSE,ME,RMSE_min,accuracy)
            
                summary_train_loss.append(train_loss)
                summary_eval_loss.append(val_loss)
                summary_RMSE.append(RMSE)
                summary_ME.append(ME)
                summary_accuracy.append(accuracy)


    except KeyboardInterrupt:
        print 'stopped'

    finally:
        # Plot the points using matplotlib
        plt.plot(range(len(summary_train_loss)), summary_train_loss)
        plt.plot(range(len(summary_eval_loss)), summary_eval_loss)
        plt.xlabel('Epochs')
        plt.ylabel('Cross-entropy')
        plt.title('Loss curve')
        plt.legend(['Train', 'Validate'])
        plt.savefig('train_val.png')
        plt.close()
#        plt.show()


        plt.plot(range(len(summary_RMSE)), summary_RMSE)
        # plt.plot(range(len(summary_ME)), summary_ME)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('RMSE')
        plt.savefig('rmse.png')
        plt.close()
        # plt.legend(['RMSE', 'ME'])
#        plt.show()

        # plt.plot(range(len(summary_RMSE)), summary_RMSE)
        plt.plot(range(len(summary_ME)), summary_ME)
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('ME')
        plt.savefig('me.png')
        plt.close()
        # plt.legend(['RMSE', 'ME'])
#        plt.show()

        plt.plot(range(len(summary_accuracy)), summary_accuracy)
        plt.xlabel('Epochs')
        plt.ylabel('Val Accuracy')
        plt.title('ME')
        plt.savefig('acc.png')
        plt.close()



        

