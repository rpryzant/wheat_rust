/*
 * this script lets you play around with 
 * cropland maps and high-res imagery
 */



// fusion table with observation data
var rawData = ee.FeatureCollection("ft:16TcyRi9YidpsXrTOS6FOM_dD7ix33P1yBKCEO5Z7");

var posLabels = ee.FeatureCollection("ft:1vs8eBqtr--Ni4qImUaAkiPJ5m2hzPwOCNF8tmXEy");
var negLabels = ee.FeatureCollection("ft:1BogvjYku8YsMd4FUSVmhULbtxLRe4TZT5oC9F2gZ");
// yeesh...
var posLabels2016 = ee.FeatureCollection("ft:1PSWb6UisupKMc9JPo93DMSmsGLorRD6xzfOdusUX");
var negLabels2016 = ee.FeatureCollection("ft:1GrfzsqJqgxvuI5AOuqSB1cdG_02U8wNX51bOO0Rb");

var posLabels2015 = ee.FeatureCollection("ft:1JpgGxCSN--3aqxv35b_I9P_ISP1PL6Kkq7P-KJRz");
var negLabels2015 = ee.FeatureCollection("ft:1aRhkvb3ehlf4mIaQ7LXz2wwGWvjfDoiY43BAivzt");

var posLabels2014 = ee.FeatureCollection("ft:1bKsBQi5y5M8KSiCPI8WeT8o8Z3XasWAEcwFh81Z7");
var negLabels2014 = ee.FeatureCollection("ft:1qJBRzLmkKeavSi_-_enxDFLu68-pj6-Gsa06evq5");

var posLabels2013 = ee.FeatureCollection("ft:1NWg3V0Ty3ogckN413_q0CtQa8sPgMOD5Se6B021A");
var negLabels2013 = ee.FeatureCollection("ft:1mne4SB_tCZwd45-pcuWQap4aMBfYl3Ew8Sp__vla");

var posLabels2012 = ee.FeatureCollection("ft:1qZAPiPniNnEVOUfWb9_S6Fbx_IPCPBkA2KfmUGS5");
var negLabels2012 = ee.FeatureCollection("ft:1ytsUIcsVT-l0-nqkQFUcIT4VwS9ImX6uBEd5C0Pu");

var posLabels2011 = ee.FeatureCollection("ft:1IW28erwWSiJUjQWAIMVtnASFul31f1cKuAhAoI5x");
var negLabels2011 = ee.FeatureCollection("ft:1uAK8UoJEhPxt-rgv-kHTeNtNm0mkUJcpUH24Vs5H");

var posLabels2010 = ee.FeatureCollection("ft:1GF2l4okoKN0Kwfz-MjmtSXDWVhJWlxEalzeuk2zS");
var negLabels2010 = ee.FeatureCollection("ft:1M1X4MZb3CaqjJNArHfpRrWVFRQaSImPtPi3uHN9Y");

var posLabels2008 = ee.FeatureCollection("ft:1-GRxOOUCN3_nGerC9WBGNdPGn3eH2Vs2xJ3mts6X");
var negLabels2008 = ee.FeatureCollection("ft:1FgvURObBkWypgLse9St0g72-3Og61hhK4r8FeCbv");

var posLabels2007 = ee.FeatureCollection("ft:1Acv8Ki9pf_llz8jk5mTAqjoJJcbqbzS_XQxnHvnv");
var negLabels2007 = ee.FeatureCollection("ft:1B_5CcEluPW5y1ofMbQv-toZEADNcxNP11SsWjMuS");





var countries = ee.FeatureCollection('ft:1tdSwUL7MVpOauSgRzqVTOwdfy17KDbw-1d9omPw');

var justEthiopia = countries.filter(ee.Filter.eq('Country', 'Ethiopia')).geometry();

var cropland = ee.Image("users/rpryzant/IIASA_IFPRI");


var ethiopiaZones = ee.FeatureCollection("ft:1R0Ax0UNacEQ7XBCmewg0QJxzxHN7ys9VMObttTTV");
print(ethiopiaZones)

    var ptszone = rawData.union();
print(ptszone)
    var selpts = ethiopiaZones.filterBounds(ptszone);
print(selpts)



// sentinel 2
    var sentinel2 = "COPERNICUS/S2";
var landsat7 = "LANDSAT/LE7_SR";


var maskCropland = function(image) {
    return image.updateMask(cropland.gt(0));
};

var sample = function(image) {
    return ee.Image(image).sampleRegions(ft);
};



// get all modis imagery, sample the pixels we're interested in
var pixels = ee.ImageCollection(landsat7)
    .filterDate('2015-07-01', '2016-02-01')
    .filterBounds(justEthiopia)
    .map(maskCropland)
    //.map(sample)
    //.flatten()
    //var image = ee.Image(pixels.first())

    //print(pixels.first())

    Map.addLayer(ethiopiaZones)
    //Map.addLayer(selpts, {color:'eb42f4'})
    // Overlay the points on the imagery to get training.
    // Map.addLayer(pixels) 


    //Map.addLayer(negLabels2007, {'color': 'FF0000'})
    //Map.addLayer(posLabels2007, {'color': '008000'})



