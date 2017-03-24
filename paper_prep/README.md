

### Requirements

* tensorflow
* sklearn
* pykml
* shapely
* numpy
* matplotlib


### Descriptions

* `main.py`: source for configuring, launching, and recording experiments
* `transfer_main.py` (incomplete): source for training model on yield data, then transfering parameters over to a rust model
* `datasets/`: survey data and region map
* `src/`
  * `analysis/`: scripts for analyzing data and results
  * `data/`: code for interacting with kml region maps, sensing data, and survey data
  * `models/`: CNN, LSTM, and linear models
  * `training/`: nested cross-validation code
  * `utils/`: msc code, mostly logging


### Usage

Each `.py` file has a brief usage description in its header.


### Prediction process

These are the steps you need to take to process some data and train models:

1. **Get the Modis Data**


