#!/usr/bin/python
import os
import ee
import sys
import time
import json
from region_featurizer import *

"""
=== DESCRIPTION
Downloads MODIS data into your google drive


=== USAGE
replace the {YOUR USERNAME} at line 51 
python modis.py ../../datasets/regions.kml ../../datasets/raw_survey.csv

"""
def appendBand(current, previous):
	previous=ee.Image(previous)
	current = current.select(bands)
	accum = ee.Algorithms.If(ee.Algorithms.IsEqual(previous,None), current, previous.addBands(ee.Image(current)))
	return accum

def mask_cropland(image):
	return image.updateMask(cropland.gt(0))

def download_image (img_collection, start, end, region, bands, folderName, fileName, taskName):
	col = img_collection.filterDate(start, end).map(mask_cropland)
	region_geom = regions.filterMetadata('name', 'equals', region).first().geometry().coordinates().getInfo()[0]
	#print region_geom
	img = col.iterate(appendBand)
	img = ee.Image(img)
	jsonTest = {
		'driveFolder':folderName,
		'driveFileNamePrefix':fileName,
		'region': region_geom,
		'scale':500,
		'maxPixels':966050808
	}
	#print jsonTest
	task = ee.batch.Export.image(img, taskName, jsonTest)
	task.start()
	while task.status()['state'] == 'RUNNING':
		print 'Running...'
	#print 'Done.', task.status()
	print fileName + ' done.'

ee.Initialize()
regions = ee.FeatureCollection('ft:133FLgnCJZsRswd2sb7Sp-od0Z90nB6P1qHuUDe57')
cropland = ee.Image("users/{YOUR USERNAME}/IIASA_IFPRI")
start = time.time()
bandsInput = raw_input("Bands(sr, tmp, gpp): ")
yearInput = int(raw_input("Year(ex. 2016): "))

sys.path.insert(0, os.path.abspath("../.."))
sf = SurveyFeaturizer(sys.argv[1], sys.argv[2])
print 'Starting region featurizer...'
seasons = {y: ('%s-06-01' % y, '%s-03-01' % (y+1)) for y in range(2007, 2017) if y == yearInput}
for season, (start, end) in seasons.iteritems():
	regions_data = sf.surveyed_regions(season)
	#print 'data_%s = %s' % (str(season), str([[r, start, end, season] for r in regions_data]))
	exec('year = %s' % (str([[r, start, end, season] for r in regions_data])))

print 'Completed.'

if bandsInput == "sr":
	bands = [0,1,2,3,4,5,6]
	collection  = ee.ImageCollection('MODIS/MOD09A1')
if bandsInput == "tmp":
	bands = [0, 4]
	collection = ee.ImageCollection('MODIS/MYD11A2')
if bandsInput == "gpp":
	bands = [0]
	collection = ee.ImageCollection('MODIS/006/MYD17A2H')

print 'Downloading for ' + str(yearInput) + '...'

for year_data in year:
	region      = year_data[0]
	start       = year_data[1]
	end         = year_data[2]
	season      = str(year_data[3])
	folder_name = 'gross_photosynthesis-' + season
	file_name   = region;

	download_image(collection, 
					start,
					end,
					region,
					bands,
					folder_name,
					file_name,
					season + '-' + file_name)

