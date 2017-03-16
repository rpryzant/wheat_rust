"""
vizualization stuff

python viz.py ../../../data/training/score_binary-histogram_data.npz

"""
import matplotlib.pyplot as plt
import sys
import numpy as np






filename = sys.argv[1]
content = np.load(filename)

# load and processes data
n_timeseries = 35
images = content['examples']
labels = content['labels']
filtered_indices = np.array([i for i, x in enumerate(images) if len(images[i]) == n_timeseries])
images = np.array([i for i in images[filtered_indices]])
labels = np.array(labels[filtered_indices])

N = len(images)

# normalize
dim = images.shape
concat = np.reshape(images, (-1, dim[2], dim[3]))   # concatenate images for each timeseries
means = np.mean(concat, axis=0)
stds = np.std(concat, axis=0)
for i in range(len(images)):
    images[i] = (images[i] - means) / (stds + 1e-6)


positives = images[np.where(labels > 0)]
negatives = images[np.where(labels == 0)]

mean_pos_hists = np.mean(positives, axis=0)   # time, bands, buckets
mean_pos_hists = np.transpose(mean_pos_hists, [1, 0, 2])  # bands, time, buckets

mean_neg_hists = np.mean(negatives, axis=0)
mean_neg_hists = np.transpose(mean_neg_hists, [1, 0, 2])



def plot_hist(band, type, data):
    dim = data.shape
    plt.subplot(10, 2, band+1)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues, alpha=0.8, linewidths=0.0)
    fig = ax.get_figure()
    plt.axis([0, dim[1], 0, dim[0]])
    plt.title('%s: band %s' % (type, band))
    plt.xlabel('buckets')
    plt.ylabel('time')
    fig.savefig("%s-histogram-%s.png" % (type, band))


for i in range(len(mean_neg_hists)):
    print 'plots for band', i
    plot_hist(i, 'negative', mean_neg_hists[i] / np.sum(mean_neg_hists[i]))
    plot_hist(i, 'positive', mean_pos_hists[i] / np.sum(mean_pos_hists[i]))

