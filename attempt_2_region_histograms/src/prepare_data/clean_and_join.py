"""

=== USAGE
python clean_and_join.py ~/Google\ Drive/ ~/Desktop/test/

"""

import numpy as np
import re
import sys
import os
import gdal
from joblib import Parallel, delayed


def gen_img_paths(gdrive_root):
    sr_root = gdrive_root + 'sr/'
    temp_root = gdrive_root + 'temp/'

    for season in range(2007, 2017):
        if season == 2009: continue
        
        sr_season = sr_root + 'reflectance-%s/' % season
        temp_season = temp_root + 'temperature-%s/' % season

        for img in os.listdir(sr_season):
            if not img.endswith('.tif'): continue

            sr_path = sr_season + img
            temp_path = temp_season + img
            # TODO - RE-DOWNLOAD THESE MISSING SHITS!!
            if not os.path.exists(temp_path):
                continue           

            yield sr_path, temp_path


def merge_image(a, a_nb, b, b_nb):
    assert (a.shape[2] / a_nb) == (b.shape[2] / b_nb), '%s, %s' % (a.shape, b.shape)
    # set up merged image
    m_nb = a_nb + b_nb                               
    m = np.zeros( (a.shape[0], a.shape[1], a.shape[2] + b.shape[2]) )
    # step though component images, adding their bands to the merger as you go
    for img_i in range(a.shape[2]/a_nb):
        ai = img_i * a_nb
        a_img = a[:,:,ai:ai+a_nb]         # pull out bands from img i of a

        bi = img_i * b_nb
        b_img = b[:,:,bi:bi+b_nb]         # bands from img i of b

        mi = img_i * (a_nb + b_nb)        # concat their bands and inject into the merger
        m[:,:,mi:mi+m_nb] = np.concatenate( (a_img, b_img), axis=2)

    return m


def preprocess_and_save(sr, temp, out):
    def metadata_from_path(path):
        return os.path.basename(path)[:-4], int(re.findall('20\d\d', path)[0])

    def read_tif(path):
        raster = gdal.Open(path).ReadAsArray()            # shape is (nbands, x, y)
        raster = np.transpose(raster, axes=(1, 2, 0))  # shape is (x, y, nbands)
        return raster

    sr_arr = read_tif(sr)
    temp_arr = read_tif(temp)
    merged = merge_image(sr_arr, 7, temp_arr, 2)

    region, season = metadata_from_path(sr)
    out_base = '%s/%s/' % (out, season)
    if not os.path.exists(out_base): os.mkdir(out_base)

    filename = '%s/%s-%s.npy' % (out_base, region, season)
    np.save(filename, merged)
    print filename, ' written'


if __name__ == '__main__':
    gdrive = sys.argv[1]
    out = sys.argv[2]

    if not os.path.exists(out):
        os.mkdir(out)

    Parallel(n_jobs=4)(delayed(preprocess_and_save)(sr, temp, out) for (sr, temp) in gen_img_paths(gdrive))

