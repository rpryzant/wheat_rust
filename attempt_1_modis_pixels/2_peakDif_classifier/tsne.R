setwd('~/Dropbox/school/ermon_lab/2_peakDif_classifier')


data = read.csv("threshold_2_stemStripe.csv")
# shuffle rows
data = data[sample(nrow(data)),]




library(readr)
library(Rtsne)


tsne <- Rtsne(train[,-1], dims = 2, perplexity=40, verbose=TRUE,check_duplicates = FALSE, max_iter = 2000)
# visualizing
colors = rainbow(length(unique(train$diseased)))
names(colors) = unique(train$diseased)
plot(tsne$Y, t='n', main="tsne")
text(tsne$Y, labels=train$diseased, col=colors[train$diseased + 1])


