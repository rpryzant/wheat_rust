"""
1) pull the raw data from google drive
2) pull the labels from a local spreadsheet
3) parse those data into the following mapping:
    {year: {location: [pixels]} }
4) pair each label with a list of all the pixels for that year and location
   a) sort pixels by date
5) write pairings to disk


=== USAGE
python generate_dataset.py ../data/threshold_2_all3.csv ~/Google\ Drive/ test.npy

"""
import pandas as pd
from datetime import datetime
import sys
import os
import sys
import re
import collections
from tqdm import tqdm
import numpy as np





def build_pixel_mapping(root):
    """ builds {year: {location: [pixels]}} mapping
    """
    d = {}
    for dir in tqdm(os.listdir(root)):
        # skip subdirectories that don't look like pixel_data_20XX and the year 2009 (too much misisng data)
        if not re.match('pixel_data_20\d\d', dir):
            continue
        year = int(re.findall('20\d\d', dir)[0])
        if year == 2009:
            continue

        # add rows from that file
        d[year] = collections.defaultdict(list)
        df = pd.read_csv(root + dir + '/pixel.csv')
        for row in iterate_rows(df):
            d[year][row['Location ID']].append(row)

    return d


def iterate_rows(df):
    """iterates through rows of a pandas df
    """
    for rowi in tqdm(range(len(df))):
        yield df.iloc[rowi,:]

def ts_from_str(s):
    """ makes datetime timestamp from string like '11/12/12 0:00' 
    """
    return datetime.strptime(re.findall('\d\d?/\d\d?/\d\d', s)[0], '%x')

def reflectance(x):
    """ extracts surface reflectance measurements (bands 1 - 7)
        from a pandas pixel row
    """
    return x.values[20:-1]



##### 1, 2, and 3: pull in data, build pixel mapping
labels_df = pd.read_csv(sys.argv[1])
modis_data = build_pixel_mapping(sys.argv[2])
outfile = sys.argv[3]


##### 4: pair labels and sequences, sort by time
data = []

for row in iterate_rows(labels_df):
    label = row.Diseased
    row_ts = ts_from_str(row.ObsDate)
    if row_ts.year == 2009:
        continue
    sequence = modis_data[row_ts.year][row['Location ID']]
    if not sequence:
        continue
    sequence = sorted(sequence, key=lambda x: x.DayOfYear)
    sequence = [[x.DayOfYear, reflectance(x)] for x  in sequence]
    data.append( [label, sequence] )

#### 5: write to disk
np.save(outfile, np.array(data))
