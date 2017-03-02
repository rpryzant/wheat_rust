"""
dataset container for output of clean_join_histogram_label

"""
import numpy as np

class Dataset():
    def __init__(self, filename):
        content = np.load(filename)
        self.ids = content['ids']
        images = content['examples']
        labels = content['labels']

        # TODO - you shouldn't have to do this!!!
        n_timeseries = 35
        filtered_indices = np.array([i for i, x in enumerate(images) if len(images[i]) == n_timeseries])  # because ragged arrays :/

        images = np.array([i for i in images[filtered_indices]])
        labels = np.array(labels[filtered_indices])

        # load images, then
        #   -- only take stacks with complete timeseries info
        #   -- subtract off mean per-band histogram
        #   -- divide by sd per feature per per-band histogram
        #   -- transpose each stack to get it in shape (buckets, time, bands)
        #          (that's what the model expects)
        dim = images.shape
        concat = np.reshape(images, (-1, dim[2], dim[3]))   # concatenate images for each timeseries
        means = np.mean(concat, axis=0)
        stds = np.std(concat, axis=0)
        for i in range(len(images)):
            images[i] = (images[i] - means) / (stds + 1e-6)
        images = np.transpose(images, (0, 3, 1, 2))   

        self.data = [(x, y) for x, y in zip(images, labels)]
        self.indices = np.arange(len(self.data))


    def get(self, i):
        return self.data[i]


    def get_data(self):
        return self.data

    def get_labels(self):
        return zip(*self.data)[1]



class DataIterator():
    def __init__(self, dataset):
        self.data = dataset.get_data()
        self.N = len(self.data)
        self.indices = np.arange(self.N)



    def xval_split(self, n_splits):
        chunk_size = len(self.data) / n_splits
        for i in self.indices[::chunk_size]:
            if i + chunk_size > self.N: continue

            pivot_batch = self.data[i: i + chunk_size]
            remainder = self.data[:i] + self.data[i + chunk_size:]

            yield pivot_batch, remainder

    def batch_iter(self, batch_size):
        i = 0
        while i + batch_size < self.N:
            yield self.data[i: i + batch_size]
            i += batch_size





