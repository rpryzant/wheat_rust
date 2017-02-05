setwd("~/Dropbox/school/ermon_lab/3_rnn_classifier/results")


#df = read.csv('overfitting_curves_raw_seqs_100_examples.csv')
df = read.csv('overfitting_curves_raw_seqs_750_examples.csv')
#df = read.csv('overfitting_curves_raw_seqs_ALL_examples.csv') # random freqs 0.699, random perms 0.623   
#df = read.csv('overfitting_curves_normed_deltas_ALL_examples.csv')   # random freq 0.694, random perm 0.576
#df = read.csv('overfitting_curves_deltas_ALL_examples.csv')         # r freq:  0.698, r perm: 0.576
df = read.csv('test.csv')


par(mfrow=c(2,1),oma = c(0, 0, 2, 0)) 
# plot acc
plot(range(df$epoch), range(df$train_acc), ylab="% accuracy", xlab="epoch", main="train/test accuracy")
#abline(h = 0.698, lty = 2)    # random frequency baseline
#abline(h = 0.576, lty = 2)    # random permutation baseline
lines(df$train_acc~df$epoch, lwd=2)
lines(df$val_acc~df$epoch, col="red", lwd=2)
# plot loss
plot(range(df$epoch), range(df$mean_loss), ylab="mean example loss", xlab="epoch", main="loss")
lines(df$mean_loss~df$epoch, lwd=2)
# main title
mtext("Raw Pixel Deltas", outer = TRUE, cex = 2.5)

