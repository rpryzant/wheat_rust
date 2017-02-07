/*
 * this script is a minimal example of pulling modis photos, smushing them together, 
 * and cropping the resulting meta-photo to fit a region
 */


// modis 500m surface reflectance
var modis_str = 'MODIS/MOD09A1';
// make an image collection out of it 
var modis_collection = ee.ImageCollection('MODIS/MOD09A1');
// open up regions fusion table
var regions = ee.FeatureCollection("ft:133FLgnCJZsRswd2sb7Sp-od0Z90nB6P1qHuUDe57");

// make cropland mask
var cropland = ee.Image("users/rpryzant/IIASA_IFPRI");
var maskCropland = function(image) {
    return image.updateMask(cropland.gt(0));
};

// apply cropland mask and time delta to modis images
modis_collection = modis_collection.filterDate('2015-11-01','2015-12-31').map(maskCropland);

// smush modis images together into one meta-image (one image per band)
var appendBand = function(cur, prev) {
    prev = ee.Image(prev);
    cur = cur.select([0,1,2,3,4,5,6]);
    var accum = ee.Algorithms.If(ee.Algorithms.IsEqual(prev,null), cur, prev.addBands(ee.Image(cur)))
    return accum;
};
var img = modis_collection.iterate(appendBand);

// pull out a region
var region = regions.filterMetadata('name', 'equals', 'AMBO_ZURIA');

// clip modis to that region and display
Map.setCenter(37.9360961914062,9.11186790466309, 10);
img = ee.Image(img)
    Map.addLayer(img.clip(region))







