"""
python histogram_and_label.py ../../data/regions.kml ../../data/raw_survey.csv ~/Desktop/test
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
            yield filepath, int(season), img[:-4]

def split_by_image(img):
    return np.split(img, img.shape[2] / BANDS_PER_IMG, axis=2)

def histogram_img(img, sr_bins, temp_bins):
    def get_bins(i):
        return sr_bins if i < 8 else temp_bins

    out = [ np.histogram(img[:,:,i], get_bins(i), density=False)[0] for i in range(img.shape[2]) ]
    out = np.array(out)
    return out

def histogram_ts(timeseries, sr_bins, temp_bins):
    return np.array( [histogram_img(img, sr_bins, temp_bins) for img in timeseries] )



def main():
    regions_kml = sys.argv[1]
    survey = sys.argv[2]
    data_root = sys.argv[3]
    sr_bins = np.linspace(1, 4999, 33)    # TODO - VERIFY?
    temp_bins = np.linspace(13000, 17000, 33) # TODO - VERIFY?

    print 'building survey features...'
    start = time.time()
    sf = SurveyFeaturizer(regions_kml, survey)
    print 'done. took {:.2f} seconds'.format(time.time() - start)

    for path, season, region in iter_img_paths(data_root):
        label = sf.label(region, season)
        timeseries = split_by_image(np.load(path))
        hist_timeseries = histogram_ts(timeseries, sr_bins, temp_bins)
        print label
        print hist_timeseries.shape

        print







if __name__ == '__main__':
    main()
