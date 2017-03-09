/*
 * you can investigate individual (lat, lon) locations with this script
 */



var pt = [38.12721, 8.64446];

var modis_sr = 'MODIS/MOD09A1';

var cropland = ee.Image("users/rpryzant/IIASA_IFPRI");


// This function masks everything but cropland.
var cropMask = function(image) {
    return image.updateMask(cropland.gt(0));
};

var landcover_collection = ee.ImageCollection(modis_sr)
    .filterBounds(ee.Geometry.Point(pt))
    .filterDate('2010-08-01', '2015-08-30')
    .map(cropMask);

var img = ee.Image(landcover_collection.first());

Map.addLayer(img, {bands: ['sur_refl_b01'], min: 0.5, max: 1, palette: ['00FFFF', '0000FF']});

var modis_reflectance = 'MODIS/MOD09A1'; // modis 500m sr

var collection = ee.ImageCollection(modis_reflectance)
    .filterBounds(ee.Geometry.Point(pt))
    .filterDate('2015-08-01', '2015-08-30');

var crop = ee.Geometry.Point(pt).buffer(500);


Map.setCenter(pt[0], pt[1], 14);

Map.addLayer(collection.mosaic().clip(crop));





