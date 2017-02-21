"""

python train.py ../../data/training/score_binary-histogram_data.npz classification

TODO
    - tidy up image loading
    - only take **first X images from each timeseries** (looks like thats what he did)
    - data augmentation thing
    - model saving/checkpointing
    - plot acc, f1
    - make regression datasets as well
    - refactor EVERYTHING
    - 
"""
from model import *
import logging
import sys
import numpy as np

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

    images = images[:200]
    labels = labels[:200]

    locations = content['ids']
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    N = len(images)
    # load images, then
    #   -- only take images with complete timeseries info
    #   -- subtract off mean per-band histogram
    #   -- transpose each image to get it in shape (buckets, photos, bands)
    #          (that's what the model expects)
    n_bands = images[0].shape[1]
    n_buckets = images[0].shape[2]

    sum_per_band = np.zeros((n_bands, n_buckets))
    for x in images:
        sum_per_band += np.sum(x, axis=0)    # sum over timeseries
    mean_per_band = sum_per_band * 1.0 / N
    for i in indices:
        images[i] = images[i] - mean_per_band
    images_new = np.zeros((N, 32, 35, 10))
    for i in indices:
        images_new[i] = np.transpose(images[i], (2, 0, 1))
    images = images_new

    train_indices = indices[:(N-(N/8))]

    val_indices = indices[(N-(N/8)):]
    val_images = images[val_indices]
    val_labels = labels[val_indices]

    print 'DATA DONE. TRAINING SET SIZE: %s TEST SET: %s' % (len(train_indices), len(val_indices))

    print 'INITIALIZING MODEL'

    model= NeuralModel(config,'net', task_type)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.22)

    # Launch the graph.
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())

    print 'MODEL DONE.'


    summary_train_loss = []
    summary_eval_loss = []
    summary_RMSE = []
    summary_ME = []
    summary_accuracy = []

    train_loss=0
    val_loss=0
    val_prediction = 0
    val_deviation = np.zeros([config.B])
    # #########################
    # block when test
    # add saver
    saver=tf.train.Saver()
    # Restore variables from disk.
    print 'ATTEMTING RESTORE' 
    try:
        saver.restore(sess, config.save_path+str(predict_year)+"CNN_model.ckpt")
    # Restore log results
        npzfile = np.load(config.save_path + str(predict_year)+'result.npz')
        summary_train_loss = npzfile['summary_train_loss'].tolist()
        summary_eval_loss = npzfile['summary_eval_loss'].tolist()
        summary_RMSE = npzfile['summary_RMSE'].tolist()
        summary_ME = npzfile['summary_ME'].tolist()
        print("Model restored.")
    except:
        print 'No history model found'
    # #########################
    
#    for i in range(config.train_step):
#        batch_index = np.random.choice(index_train, size=config.B)
#        x_batch = image_all[batch_index]
#        y_batch = yield_all[batch_index]
#        _, train_loss = sess.run([model.train_op, model.loss_err], feed_dict={
#            model.x: x_batch,
#            model.y: y_batch,
#            model.lr: config.lr,
#            model.keep_prob: config.drop_out
#            })
#        print train_loss


    print 'TRAINING...'
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



        

