

## Requirements

* tensorflow
* sklearn
* pykml
* shapely
* numpy
* matplotlib


## Where Everything Is

* `main.py`: source for configuring, launching, and recording experiments
* `transfer_main.py` (incomplete): source for training model on yield data, then transfering parameters over to a rust model
* `datasets/`: survey data and region map
* `src/`
  * `analysis/`: scripts for analyzing data and results
  * `data/`: code for interacting with kml region maps, sensing data, and survey data
  * `models/`: CNN, LSTM, and linear models
  * `training/`: nested cross-validation code
  * `utils/`: msc code, mostly logging



## How to Run the Pipeline

These are the steps you need to take to process some data and train models:

1. **Get the IIASA-IFPRI Cropland Map**
    * make a google earth engine account: https://earthengine.google.com/
    * Go to http://cropland.geo-wiki.org/downloads/ and download the http://cropland.geo-wiki.org/downloads/ (its the first download link).
    * Install GDAL. Go to http://tilemill-project.github.io/tilemill/docs/guides/gdal/ and follow the directions for your machine/OS.
    * Unzip the downloaded cropland map, cd into that directory, and convert the `.img` to a `.tif`:
```
unzip cropland_hybrid_14052014v8.zip -d .
cd cropland_hybrid_14052014v8
gdal_translate -of GTiff Hybrid_14052014V8.img cropland_map.tif
```
    * Upload the `.tif` to google earth engine. Go to https://code.earthengine.google.com/, then click on the "assets" tab. Click "NEW", then "Image Upload". Select your recently converted `.tif`, then click ok.
    * An "Upload" job should be triggered in your "task" panel. Once that's complete the map should be in your "assets" tab.


## Usage

Each `.py` file has a brief usage description in its header.



