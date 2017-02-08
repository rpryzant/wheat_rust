"""



=== USAGE
python pull_MODIS_per_region.py ../../data/regions.kml ../../data/raw_survey.csv

"""
import os
import ee
import sys
sys.path.insert(0, os.path.abspath("../.."))
from src.label_regions.region_utils import *







sf = SurveyFeaturizer(sys.argv[1], sys.argv[2])

ee.Initialize()


regions = ee.FeatureCollection("ft:133FLgnCJZsRswd2sb7Sp-od0Z90nB6P1qHuUDe57")
MODIS_collection = ee.ImageCollection('MODIS/MOD09A1')


MODIS_collection.filterDate('2001-12-31','2015-12-31')

region = regions.filterMetadata('name', 'equals', 'ADDIS_KETEMA')

region = ee.Feature(region.first())

export_oneimage(MODIS_collection.clip())




