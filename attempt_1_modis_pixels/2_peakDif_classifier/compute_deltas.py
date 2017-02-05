"""
1) Get pixel at peak greens (near infared / red) for that year
2) get pixel one month later
3) get delta pixel
4) regress on that 


=== USAGE
python compute_deltas.py ../data/data_i_need_threshold2.csv ~/Google\ Drive/ threshold_1_stem_stripe.csv
"""

import sys
import pandas as pd
from datetime import datetime
import os
import re
import collections
from tqdm import tqdm


def write(loc, s):
    """ writes a string s to location loc
    """
    output = open(loc, 'w')
    output.write(s)
    output.close()


def ts_from_str(s):
    """ makes datetime timestamp from string like '11/12/12 0:00'
    """
    return datetime.strptime(re.findall('\d\d?/\d\d?/\d\d', s)[0], '%x')


def iterate_rows(df):
    """iterates through rows of pandas df
    """
    for rowi in tqdm(range(len(df))):
        yield df.iloc[rowi,:]


def build_pixel_mapping(root):
    """ builds {year: {location: [pixels]}} mapping
    """
    d = {}
    for dir in tqdm(os.listdir(root)):
        if not re.match('pixel_data_20\d\d', dir):
            continue
        year = int(re.findall('20\d\d', dir)[0])

        if year == 2009:        # skip 2009: too much missing data
            continue

        d[year] = collections.defaultdict(list)
        
        df = pd.read_csv(root + dir + '/pixel.csv')
        for row in iterate_rows(df):
            d[year][row['Location ID']].append(row)
    return d


####### 1: build pixel mapping
df = pd.read_csv(sys.argv[1])
drive_root = sys.argv[2]
output = sys.argv[3]


PIXEL_MAPPING = build_pixel_mapping(drive_root)

####### 2: compute deltas
outgoing_rows = [['diseased', 'sur_refl_b01', 'sur_refl_b02','sur_refl_b03', 'sur_refl_b04', 'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07']]

for row in iterate_rows(df):
    row_ts = ts_from_str(row.ObsDate)
    row_year = row_ts.year
    row_loc = row['Location ID']

    if row_year == 2009:    # skip 2009: too much missing data
        continue

    # sort pixels for this location & year by day of year
    modis_pixels = sorted(PIXEL_MAPPING[row_year][row_loc], key=lambda x: x.DayOfYear)
    if not modis_pixels:
        continue

    # pull out the pixel with peak greenery (measured with IR (band 2) / R (band 1))
    _, max_green_i = max((x.sur_refl_b02 * 1.0 / x.sur_refl_b01, i) for i, x in enumerate(modis_pixels))
    max_green = modis_pixels[max_green_i]

    # pull out first pixel that's a month later
    min_green = None
    for x in modis_pixels[max_green_i:]:
        min_green = x
        if abs(x.DayOfYear - max_green.DayOfYear) > 24:
            break

    delta = min_green[20:-1] - max_green[20:-1]

    outgoing_rows.append([row.Diseased] + list(delta))

        
##### 3: write data


s = '\n'.join(','.join(str(attr) for attr in row) for row in outgoing_rows)

write(output, s)







