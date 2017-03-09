"""
=== DESCRIPTION
This file generates region listings found at the top of pull_modis.js

=== USAGE
python generate_region_listings.py ../../../data/regions.kml ../../../data/raw_survey.csv
"""
import os
import ee
import sys
sys.path.insert(0, os.path.abspath("../../.."))
from src.prepare_data.region_featurizer import *
import time





print 'building survey features...'
start = time.time()
sf = SurveyFeaturizer(sys.argv[1], sys.argv[2])
print 'done. took {:.2f} seconds'.format(time.time() - start)

seasons = {y: ('%s-06-01' % y, '%s-03-01' % (y+1)) for y in range(2007, 2017) if y != 2009}
for season, (start, end) in seasons.iteritems():
    print
    print
    regions = sf.surveyed_regions(season)
    print 'var data_%s = %s;' % (str(season), str([[r, start, end, season] for r in regions]))


