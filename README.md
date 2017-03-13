

# Description

At a high level, this project tries to train machine learning models on satellite imagery in order to discover fungal outbreaks among Ethiopian wheat fields. 

The pathogen we're concerned with is named *Puccinia triticina* and is commonly referred to as wheat rust. It comes in three types (yellow, stripe, leaf) and looks like this:

<img src="img/rust.png" width="200">

### Data

For labels, we have close to 10,000 field-level observations taken over the course of a decade. These surveys were conducted by the folks at [RustTracker.org](http://rusttracker.cimmyt.org/). Each observation has a latitude, longitude, date, and infection severity rating for three strains of our pathogen. Below is a subset of the data for the 2010 growing season.

![](img/survey.png)

For examples, we are using photos taken from two NASA spacecraft: Terra and Aqua. These photos come from a product called MODIS. We selected this product due to its high temporal resolution (8 days). To reduce the dimensionality of this imagery, we train models on histograms of pixel frequencies as one moves through time. Example histograms for healthy (left) and diseased (right) fields are below:

![](img/hist.png)


### Model

To learn on these data, we experimented with a variety of linear and nonlinear classifiers. As of 3/5/17, it appears that deep learning models that cook up their own representations of the data work best. These include 
* Recurrent Neural Networks & LSTMS
* Feed-Forward Nerual Networks
* Convolutional Neural Networks. 




# Cropland Mask Directions

These are the steps you need to do to get the IIASA-IFPRI Cropland Map into google earth engine:

1. Go to http://cropland.geo-wiki.org/downloads/ and download the http://cropland.geo-wiki.org/downloads/ (its the firt download link).
2. Install GDAL. Go to http://tilemill-project.github.io/tilemill/docs/guides/gdal/ and follow the directions for your machine/OS.
3. Unzip the downloaded cropland map, cd into that directory, and convert the `.img` to a `.tif`:
```
unzip cropland_hybrid_14052014v8.zip -d .
cd cropland_hybrid_14052014v8
gdal_translate -of GTiff Hybrid_14052014V8.img cropland_map.tif
```
4. Upload the `.tif` to google earth engine. Go to https://code.earthengine.google.com/, then click on the "assets" tab. Click "NEW", then "Image Upload". Select your recently converted `.tif`, then click ok. 
5. An "Upload" job should be triggered in your "task" panel. Once that's complete the map should be in your "assets" tab.

