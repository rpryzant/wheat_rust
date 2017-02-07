


import ee






ee.Initialize()


regions = ee.FeatureCollection("ft:133FLgnCJZsRswd2sb7Sp-od0Z90nB6P1qHuUDe57")

MODIS_collection = ee.ImageCollection('MODIS/MOD09A1')


MODIS_collection.filterDate('2001-12-31','2015-12-31')



region = regions.filterMetadata('name', 'equals', 'ADDIS_KETEMA')

region = ee.Feature(region.first())






