"""
local: 
python histogram_and_label.py ../../data/regions.kml ../../data/raw_survey.csv ~/Desktop/test ~/Desktop/test2
global:
python histogram_and_label.py ../../data/regions.kml ../../data/raw_survey.csv ../../../../datasets/ ../../../../datasets/training/

"""
import os
import numpy as np
import sys
sys.path.insert(0, os.path.abspath("../.."))
from src.label_regions.region_featurizer import *

# 7 from sr, 2 from temp
SR_BANDS = 7
TEMP_BANDS = 2
BANDS_PER_IMG = SR_BANDS + TEMP_BANDS

def iter_img_paths(data_root):
    for season in os.listdir(data_root):
        for img in os.listdir(os.path.join(data_root, season)):
            filepath = os.path.join(data_root, season, img)
            yield filepath, int(season), img.split('-')[0]

def split_by_image(img):
    return np.split(img, img.shape[2] / BANDS_PER_IMG, axis=2)


def histogram_ts(timeseries, sr_bins, temp_bins):
    def histogram_img(img):
        def get_bins(i):
            return sr_bins if i < 8 else temp_bins

        out = [ np.histogram(img[:,:,i], get_bins(i), density=False)[0] for i in range(img.shape[2]) ]
        out = np.array(out)
        return out

    return np.array( [histogram_img(img) for img in timeseries] )


def histogram_data(data_root, sf):
    sr_bins = np.linspace(1, 4999, 33)    # TODO - VERIFY?
    temp_bins = np.linspace(13000, 17000, 33) # TODO - VERIFY?

    labels = []
    examples = []
    ids = []
    for path, season, region in iter_img_paths(data_root):
        label = sf.label(region, season)
        timeseries = split_by_image(np.load(path))
        hist_timeseries = histogram_ts(timeseries, sr_bins, temp_bins)

        labels.append(label)
        examples.append(hist_timeseries)
        ids.append('%s-%s' % (region, season))
    return np.array(labels), np.array(examples), np.array(ids)

def main():
    regions_kml = sys.argv[1]
    survey = sys.argv[2]
    data_root = sys.argv[3]
    outpath = sys.argv[4]

    print 'building survey features...'
    start = time.time()
    sf = SurveyFeaturizer(regions_kml, survey)
    print 'done. took {:.2f} seconds'.format(time.time() - start)

    print 'histogramming...'
    start = time.time()
    labels, examples, ids = histogram_data(data_root, sf)
    print 'done. took {:.2f} seconds'.format(time.time() - start)
    
    print 'writing data...'
    start = time.time()
    np.savez(outpath + '/histogram_data.npz', labels=labels, examples=examples, ids=ids)
    print 'done. took {:.2f} seconds'.format(time.time() - start)

if __name__ == '__main__':
    main()
