
/*

NOTE: START WITH OLD PULL MODIS PRELUDE

 */


/*
var target_images = data_2007.concat(data_2008)
                             .concat(data_2010)
                             .concat(data_2011)
                             .concat(data_2012)
                             .concat(data_2013)
                             .concat(data_2014)
                             .concat(data_2015)
                             .concat(data_2016);
*/


var regions = ee.FeatureCollection("ft:133FLgnCJZsRswd2sb7Sp-od0Z90nB6P1qHuUDe57");


var download_image = function(img_collection, start, end, region, bands, folderName, filename, taskname) {
    // apply cropland mask and time delta to modis images
    var col = img_collection.filterDate(start, end).map(maskCropland);

    // smush modis images together into one meta-image (one image per band)
    var appendBand = function(cur, prev) {
	prev = ee.Image(prev);
	cur = cur.select(bands);
	var accum = ee.Algorithms.If(ee.Algorithms.IsEqual(prev,null), cur, prev.addBands(ee.Image(cur)))
	return accum;
    };

    var region_geom = regions.filterMetadata('name', 'equals', region).first().geometry();

  
    var img = col.iterate(appendBand);

  
    img = ee.Image(img);



    Export.image.toDrive({
	    folder: folderName,
		fileNamePrefix: filename,
		image: img,
		description: taskname,
		scale: 500,
		maxPixels: 966050808,
		region: region_geom
		});
};




var cropland = ee.Image("users/rpryzant/IIASA_IFPRI");
var maskCropland = function(image) {
    return image.updateMask(cropland.gt(0));
};


//var bands = [0,1,2,3,4,5,6]; // sr
//var bands = [0, 4]; // temp
var bands = [0]; // gpp


for (var i = 0; i < data_2007.length; i++) {
    // reinitialize in case is mutated by filtering 
    //var collection  = ee.ImageCollection('MODIS/MOD09A1');  //reflectance
    // var collection = ee.ImageCollection('MODIS/MYD11A2'); // tmp 
    var collection = ee.ImageCollection('MODIS/006/MYD17A2H'); // pp
    var region      = data_2007[i][0];
    var start       = data_2007[i][1];
    var end         = data_2007[i][2];
    var season      = data_2007[i][3].toString()
	var folder_name = 'gross_photosynthesis-' + season;
    var file_name   = region;

    download_image(collection, 
		   start,
		   end,
		   region,
		   bands,
		   folder_name,
		   file_name,
		   season + '-' + file_name);

}

/*
var collection  = ee.ImageCollection('MODIS/MYD11A2'); 
download_image(collection, '2008-06-01', '2009-03-01', 'MESELA', [0,1,2,3,4,5,6], 'reflectance-2008', 'MESELA', '2008-MESELA')
*/
