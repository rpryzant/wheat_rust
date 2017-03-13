"""
dataset container for output of clean_join_histogram_label

"""
import numpy as np
import math 

class Dataset():
    def __init__(self, filename, config):
        self.config = config
        content = np.load(filename)
        self.ids = content['ids']
        images = content['examples']
        labels = content['labels']

        # TODO - you shouldn't have to do this!!!
        n_timeseries = 35

        if config.padding == 'true':
            print 'PADDING...'
            padded_images = []
            DEFAULT_BANDS = 10
            lengths = []
            for i, x in enumerate(images):
                lengths.append(len(x))
                if len(x) < n_timeseries:
                    x = np.append(x, np.zeros([n_timeseries - len(x), DEFAULT_BANDS, config.W]), axis=0)
                padded_images.append(x)
            filtered_indices = np.arange(len(padded_images))
            images = np.array(padded_images)
            lengths = np.array(lengths)

        else:
            print 'NOT PADDING...'
            filtered_indices = np.array([i for i, x in enumerate(images) if len(images[i]) == n_timeseries])  # because ragged arrays :/
            images = np.array([i for i in images[filtered_indices]])
            lengths = np.array([len(x) for x in images])

        labels = np.array(labels[filtered_indices])

        print len(images)
        print len(labels)

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

        if config.deletion_band < 15: 
            # TODO: there HAS to be a better way to do this
            images = np.transpose(images, [3, 0, 1, 2])     
            images = np.array([img for i, img in enumerate(images) if i != self.config.deletion_band])
            images = np.transpose(images, [1, 2, 3, 0])
        self.data = [(x, y, l) for x, y, l in zip(images, labels, lengths)]

        self.indices = np.arange(len(self.data))


    def get(self, i):
        return self.data[i]


    def get_data(self):
        return self.data

    def get_labels(self):
        return zip(*self.data)[1]


class BaselineDataset():
    # http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0093107
    def __init__(self, filename, config):
        # TODO - you shouldn't have to do this!!!
        n_timeseries = 35

        self.config = config
        print 'LOADING ', filename
        content = np.load(filename)
        print 'LOAD DONE'
        self.ids = content['ids']
        out_vectors = []
        indices = []
        for i, timeseries in enumerate(content['examples']):
            new_timeseries = []            
            if len(timeseries) < n_timeseries:
                continue
            for image in timeseries:
                R_R = np.mean(image[0])
                R_NIR = np.mean(image[1])
                R_B = np.mean(image[2])
                R_G = np.mean(image[3])
                features = [
                    R_R,
                    R_NIR,
                    R_B,
                    R_G,
                    self.SR(R_R, R_NIR),
                    self.NVDI(R_R, R_NIR),
                    self.GNVDI(R_G, R_NIR),
                    self.TVI(R_R, R_G, R_NIR),
                    self.SAVI(R_R, R_NIR),
                    self.OSAVI(R_R, R_NIR),
                    self.MSR(R_R, R_NIR),
                    self.NLI(R_R, R_NIR),
                    self.RVDI(R_R, R_NIR),
                    self.CARI(R_R, R_G, R_NIR),
                    self.PSRI(R_R, R_B, R_NIR)
                ]
                # derivatives
                if len(new_timeseries) == 0:
                    features += [0] * len(features)
                else:
                    features += list(np.array(features) - np.array(new_timeseries[-1][:15]))  # 15 features for now TODO make dynmic on first half of vectors

                new_timeseries.append(features)
            out_vectors.append(new_timeseries)
            indices.append(i)

        indices = np.array(indices)
        labels = content['labels'][indices]
        print len(labels), np.count_nonzero(np.array(labels))
        self.data = [(x, y, None) for x, y in zip(out_vectors, labels)]   # need none to fit into (example, label, length) format



    def SR(self, R_R, R_NIR):
        """ simple ratio """
        return R_NIR * 1.0 / R_R

    def NVDI(self, R_R, R_NIR):
        """ normalized difference vegetation index """
        return (R_NIR - R_R) * 1.0 / (R_NIR + R_R)

    def GNVDI(self, R_G, R_NIR):
        """ green normalized difference vegetation index """
        return (R_NIR - R_G) * 1.0 / (R_NIR + R_G)

    def TVI(self, R_R, R_G, R_NIR):
        """ triangular vegetation index """
        return 0.5 * (120 * (R_NIR - R_G) - 200 * (R_R - R_G))

    def SAVI(self, R_R, R_NIR):
        """ soil adjusted vegetation index """
        return 1.5 * (R_NIR - R_R) / (R_NIR + R_R + 1.5)

    def OSAVI(self, R_R, R_NIR):
        """ optimized soil adjusted vegetation index """
        return (R_NIR / R_R) / (R_NIR + R_R + 16)

    def MSR(self, R_R, R_NIR):
        """ modified simple ratio """
        return (R_NIR / R_R - 1) / (R_NIR / R_R + 1)**0.5

    def NLI(self, R_R, R_NIR):
        """non-linear vegetation index """
        return (R_NIR**2 - R_R) / (R_NIR**2 + R_R)

    def RVDI(self, R_R, R_NIR):
        """ re-normalized difference vegetation index """
        return (R_NIR - R_R) / (R_NIR + R_R)**0.5


    def CARI(self, R_R, R_G, R_NIR):
        """ clorophyll absorption ratio """
        a = (R_NIR - R_R) * 1.0 / 150
        b = R_G - (a * 550)

        return ((a * 670 + R_R + b) * 1.0  / math.sqrt(a**2 + 1)) * (R_NIR * 1.0 / R_R)

    def PSRI(self, R_R, R_B, R_NIR):
        """ plant senescence reflectance index """
        return (R_R - R_B) / R_NIR



# TODO - CARI
#      - PSRI

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





