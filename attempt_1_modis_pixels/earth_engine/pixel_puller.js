// fusion table with observation data
var ft = ee.FeatureCollection("ft:16TcyRi9YidpsXrTOS6FOM_dD7ix33P1yBKCEO5Z7");

// modis surface reflectance 500m
var modis_sr = "MODIS/MOD09A1";

// wrapper for pixel sampling filter
var sample = function(image) {
    return image.sampleRegions(ft);
};

// get all modis imagery, sample the pixels we're interested in
var pixels = ee.ImageCollection(modis_sr)
    .filterDate('2016-01-01', '2017-01-01')
    .map(sample)
    .flatten();
print(pixels.first());

// Export the FeatureCollection to a csv file.
Export.table.toDrive({
	collection: pixels,
	    description:'pixels2016',
	    folder: 'pixel_data_2016',
	    fileNamePrefix: 'pixel',
	    fileFormat: 'CSV'
	    });
      