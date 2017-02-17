"""
=== DESCRIPTION
This file takes a bunch of stacked up timeseries images from various satellite sources
  and combines each stack, then writes those data as numpy arrays

I used this file to combine ethiopia region-level season-long timeseries data for 
   -MODIS MOD09A1 (surface reflectance, bands 0-6)
   -MODIS MYD11A2 (temperature, bands 0 & 4)
   -MODIS MYD17A2H (gross photosynthesis, band 0)

=== PRECONDITIONS
sys.argv[1] must be a directory root "root" structured as follows:

root/
  sr/
    IMAGE1.tif
    IMAGE2.tif
    ...
  temp/
    IMAGE1.tif
    IMAGE2.tif
    ...
  gpp/
    IMAGE1.tif
    IMAGE2.tif
    ...

=== USAGE
local:
python clean_and_join.py ~/Google\ Drive/ ~/Desktop/test/
remote ( ssfhs atlas.stanford.edu:/atlas/u/rpryzant/datasets mount point ~/foreign_mount/ ):
python clean_and_join.py ~/foreign_mount/ ~/Desktop/test/


python clean_join_histogram_label.py ~/foreign_mount/ ../../data/regions.kml ../../data/raw_survey.csv ~/Desktop/test score_binary

"""

import numpy as np
import re
import sys
import os
import gdal
from joblib import Parallel, delayed
from tqdm import tqdm
import time
sys.path.insert(0, os.path.abspath("../.."))
from region_featurizer import *

SR_BANDS = 7
TEMP_BANDS = 2
GPP_BANDS = 1
BANDS_PER_IMG = SR_BANDS + TEMP_BANDS + GPP_BANDS



def gen_img_paths(gdrive_root):
    """ iterator that spits out reflectance, temperature, photosynthesis 
        images simultaneously 
    """
    sr_root = gdrive_root + 'sr/'
    temp_root = gdrive_root + 'temp/'
    gpp_root = gdrive_root + 'gpp/'

    for season in tqdm(range(2007, 2017)):
        if season == 2009: continue
        
        sr_season = sr_root + 'reflectance-%s/' % season
        temp_season = temp_root + 'temperature-%s/' % season
        gpp_season = gpp_root + 'gross_photosynthesis-%s/' % season

        for img in tqdm(os.listdir(sr_season)):
            if not img.endswith('.tif'): continue

            sr_path = sr_season + img
            temp_path = temp_season + img
            gpp_path = gpp_season + img
            # TODO - RE-DOWNLOAD THE MISSING SHITS!!
            if not os.path.exists(temp_path) or not os.path.exists(gpp_path):
                continue           

            yield sr_path, temp_path, gpp_path




def preprocess(sr, temp, gpp):
    """ convert sr/temp/gpp images to np and merge them

        @return: timeseries, a np array with shape (num images, width, height, num stacked bands)
    """
    def merge_image(a, a_nb, b, b_nb, c, c_nb):
        """ given np arrays of 3 image stacks all with shape (x, y, num bands * num images) and
            the number of bands per image, interleave their component images 
        """
        assert (a.shape[0] == b.shape[0] == c.shape[0] and a.shape[1] == b.shape[1] == c.shape[1]), \
            "merger of images with different sizes! \n\ta: %s \n\tb: %s" % (a.shape, b.shape, c.shape)

        # set up merged image
        m_nb = a_nb + b_nb + c_nb               
        timeseries = []
        m = np.zeros( (a.shape[0], a.shape[1], a.shape[2] + b.shape[2] + c.shape[2]) )
        # step though component images, adding their bands to the merger as you go
        for img_i in range(a.shape[2]/a_nb):
            ai = img_i * a_nb
            a_img = a[:,:,ai:ai+a_nb]         # pull out bands from img i of a

            bi = img_i * b_nb
            b_img = b[:,:,bi:bi+b_nb]         # bands from img i of b

            ci = img_i * c_nb
            c_img = c[:,:,ci:ci+c_nb]         # bands from img i of c

            # guard against prematurely exausting one source (seems like gpp is slightly staggered)
            if not (a_img.shape[2] > 0 and b_img.shape[2] > 0 and c_img.shape[2] > 0):
                print 'EARLY EXIT: stopping at image', img_i
                break

            timeseries.append(np.concatenate( (a_img, b_img, c_img), axis=2))

        return np.array(timeseries)

    def read_tif(path):
        raster = gdal.Open(path).ReadAsArray()            # shape is (nbands, x, y)
        raster = np.transpose(raster, axes=(1, 2, 0))  # shape is (x, y, nbands)
        return raster

    sr_arr = read_tif(sr)
    temp_arr = read_tif(temp)
    gpp_arr = read_tif(gpp)
    timeseries = merge_image(sr_arr, SR_BANDS, temp_arr, TEMP_BANDS, gpp_arr, GPP_BANDS)

    return timeseries


def metadata_from_path(path):
    return os.path.basename(path)[:-4], int(re.findall('20\d\d', path)[0])




def histogram_stacked_image(timeseries):
    sr_bins = np.linspace(1, 4999, 33)    # TODO - VERIFY?
    temp_bins = np.linspace(13000, 17000, 33) # TODO - VERIFY?
    gcc_bins = np.linspace(1, 999, 33)

    def get_bins(i):
        if i in range(0, 7):
            return sr_bins
        elif i in range(7, 9):
            return temp_bins
        else:
            return gcc_bins

    def histogram(x):
        out = [ np.histogram(x[:,:,i], get_bins(i), density=False)[0] for i in range(x.shape[2]) ]
        return np.array(out)

    return np.array( [histogram(x) for x in timeseries] )

    


if __name__ == '__main__':
    gdrive = sys.argv[1]
    regions_kml = sys.argv[2]
    survey = sys.argv[3]
    out = sys.argv[4]
    label_type = sys.argv[5]

    if not os.path.exists(out):
        os.mkdir(out)
        
    print '========================================'
    print 'building survey features...'
    start = time.time()
    sf = SurveyFeaturizer(regions_kml, survey, Thresholders.maxStemStripe3Leaf)
    print 'done. took {:.2f} seconds'.format(time.time() - start)

    print '======================================='
    print 'processing images...'
    labels = []
    examples = []
    ids = []
    skipped = []
    for (sr, temp, gpp) in gen_img_paths(gdrive):
        try:
            region, season = metadata_from_path(sr)
            print 'START: %s-%s' % (region, season)

            print '\t merging data sources...'
            start = time.time()
            merged_timeseries = preprocess(sr, temp, gpp)
            print '\t done! Took {:.2} seconds'.format(time.time() - start)

            print 
            print '\t histogramming...'
            start = time.time()
            hist = histogram_stacked_image(merged_timeseries)
            print '\t done! took {:.2f} seconds'.format(time.time() - start)

            labels.append(sf.label(region, season, type=label_type))
            examples.append(hist)
            ids.append('%s-%s' % (region, season))

            print
            print '= = = = = = = = = = = = = = = = = ='

        except Exception as e:
            skipped.append( (region, season) )
            print e
            print 'SOMETHING BROKE!!!!!!!!!!!!!1'


    print '====================================='
    print 'writing data to %s...' % out
    start = time.time()
    if not os.path.exists(out): os.mkdir(out)
    np.savez(out + '/histogram_data.npz', labels=labels, examples=examples, ids=ids)    
    print 'done! took {:.2f} seconds. wrote {} examples. skipped {}.'.format(time.time() - start, len(examples), len(skipped))
    print 'skipped: '
    for x in skipped:
        print x
