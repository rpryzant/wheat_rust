"""
I'm downloading .tif images into google drive. The pipeline is fairly brittle, so this is a 
quick script that thumbs though the files I have and spits out the params needed to fill 
in the gaps

python find_missing_files.py ../../data/regions.kml ../../data/raw_survey.csv /Users/rapigan/Google\ Drive/reflectance-2007


"""

import ee
import os
import sys
import time
import re
import string
from collections import defaultdict
sys.path.insert(0, os.path.abspath("../.."))
from src.label_regions.region_featurizer import *





# get survey features
print 'building survey features...'
start = time.time()
sf = SurveyFeaturizer(sys.argv[1], sys.argv[2])
print 'done. took {:.2f} seconds'.format(time.time() - start)

# use features to build library of images that should have been downloaded
images = {}
seasons = {y: ('%s-06-01' % y, '%s-03-01' % (y+1)) for y in range(2007, 2017) if y != 2009}
for season, (start, end) in seasons.iteritems():
    regions = sf.surveyed_regions(season)
    images[season] = {r.replace("'", ""): [r, start, end, season] for r in regions}


gdrive = sys.argv[3]
season = int(re.findall('20\d\d', gdrive)[0])

files = [x[:-4] for x in os.listdir(gdrive)]

missing_images = set(images[season].keys()) - set(files)

print [images[season][x] for x in missing_images]






