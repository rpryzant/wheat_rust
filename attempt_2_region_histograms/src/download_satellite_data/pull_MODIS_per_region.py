"""



=== USAGE
python pull_MODIS_per_region.py ../../data/regions.kml ../../data/raw_survey.csv

"""
import os
import ee
import sys
sys.path.insert(0, os.path.abspath("../.."))
from src.label_regions.region_utils import *
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



print '...'

print 'initializing earth engine connection...'
start = time.time()
ee.Initialize()
regions = ee.FeatureCollection("ft:133FLgnCJZsRswd2sb7Sp-od0Z90nB6P1qHuUDe57")

#cropland = ee.Image("/Users/rapigan/Downloads/cropland_hybrid_14052014v8/out.tif")
cropland = ee.Image("users/rpryzant/IIASA_IFPRI")
def mask_cropland(image):
    return image.updateMask(cropland.gt(0))
print 'done. took {:.2f} seconds'.format(time.time() - start)




def appender(bands):
    def _append_band(cur, prev):
        """ takes all the bands from a stack of images and combines them into one meta-image 
            with many bands
            e.g. given images I1:[band1, band2] and I2:[band1, band2], this function will return
                 Iappend:[band1, band2, band1_1, band2_1] with bandX_1 coming from I2
        """
        prev = ee.Image(prev)
        cur = cur.select(bands)
        accum = ee.Algorithms.If(ee.Algorithms.IsEqual(prev, None), cur, prev.addBands(ee.Image(cur)))
        return accum

    return _append_band


def export_image(img, folder, name, scale):
    task = ee.batch.Export.image(img, name, {
            'folder': folder,
            'fileNamePrefix': name,
            'scale': scale,
            'maxPixels': 13205041984               # todo - better value
          })
    task.start()
    print '== EXPORTING %s/%s' % (folder, name)
    while task.status()['state'] in ['RUNNING']:
        print '\t running...'
        time.sleep(10)
    print '\t done', task.status()
    

seasons = {y: ('%s-06-01' % y, '%s-03-01' % (y+1)) for y in range(2007, 2017) if y != 2009}
scale = 500     # modis is 500m resolution. downsample coarser images to this


# TODO DOESNT WORK???
MODIS_collection = ee.ImageCollection('MODIS/MOD09A1')
modis_images = MODIS_collection.filterDate('2015-06-01', '2016-03-01').map(mask_cropland)
modis_meta_image = modis_images.iterate(appender([0,1,2,3,4,5,6]))    # sr bands 1-7
region = regions.filterMetadata('name', 'equals', 'AMBO_ZURIA')
img = ee.Image(modis_meta_image)
export_image(img.clip(region), 'test_folder_2', 'test_prefix', 500)

quit() 


############  1: surface reflectance
folder = 'reflectance_%s'
MODIS_collection = ee.ImageCollection('MODIS/MOD09A1')
for season, (start, end) in seasons.iteritems():
    modis_images = MODIS_collection.filterDate(start, end).map(mask_cropland)
    modis_meta_image = modis_images.iterate(appender([0,1,2,3,4,5,6]))    # sr bands 1-7
    modis_meta_image = ee.Image(modis_meta_image)
    for region_name in sf.surveyed_regions(season):
        region = regions.filterMetadata('name', 'equals', region_name)
        export_image(modis_meta_image.clip(region), folder % season, region_name, scale)



###########   2: temperature
folder = 'temp_%s'
MODIS_collection = ee.ImageCollection('MODIS/MYD11A2')
for season, (start, end) in seasons.iteritems():
    modis_images = MODIS_collection.filterDate(start, end).map(mask_cropland)
    modis_meta_image = modis_images.iterate(appender([0,4]))             # day and night tmp
    modis_meta_image = ee.Image(modis_meta_image)
    for region_name in sf.surveyed_regions(season):
        region = regions.filterMetadata('name', 'equals', region_name)
        export_image(modis_meta_image.clip(region), folder % season, region_name, scale)



###########  3: primary productivity
folder = 'gpp_%s'
MODIS_collection = ee.ImageCollection('MODIS/006/MYD17A2H')
for season, (start, end) in seasons.iteritems():
    modis_images = MODIS_collection.filterDate(start, end).map(mask_cropland)
    modis_meta_image = modis_images.iterate(appender([0,1]))             # gpp and net photosythesis (gpp - maintenence respo)
    modis_meta_image = ee.Image(modis_meta_image)
    for region_name in sf.surveyed_regions(season):
        region = regions.filterMetadata('name', 'equals', region_name)
        export_image(modis_meta_image.clip(region), folder % season, region_name, scale)
