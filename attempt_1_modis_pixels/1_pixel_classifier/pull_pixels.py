"""


=== read from data
date = datetime.datetime.strptime(d.ObsDate[1][:-5], '%x')

=== from earth engine
date = datetime.datetime.fromtimestamp(your_timestamp / 1e3)



=== USAGE

python pull_pixels.py ../data/data_i_need.csv ~/Google\ Drive/



"""

import sys
import pandas as pd
from datetime import datetime
import re
from tqdm import tqdm
import os


DATASET_PREFIX = 'pixel_data_'

def get_DOY(date):
    """ gets the day of year for a datetime object (this is how MODIS assigns time info) """
    return date.timetuple().tm_yday

def get_pixel_from_record(locid, date, data_root):
    """ find the MODIS pixel that matches a given location and date
    """
    query_doy = get_DOY(date)       # get day of year for query
    min_delta = 1000
    closest_pixel = None
    d = pd.read_csv('%spixel_data_%s/pixel.csv' % (data_root, date.year))
    for row, (data_locid, data_doy) in enumerate(zip(d['Location ID'], d.DayOfYear)):
        if data_locid == locid:
            if abs(data_doy - query_doy) < min_delta:
                closest_pixel = row
                min_delta = abs(data_doy - query_doy)

    return d.loc[[closest_pixel]]



labels = pd.read_csv(sys.argv[1])
gee_root = sys.argv[2]




def get_closest_key(data, locid, doy):
    """ modis isn't daily, so get the closest pixel at loc "locid" to day "doy"
    """
    # knock off the last 2 chars of the id because google earth engine appended a '.0' to the end of 'em
    x =  min((abs(int(d) - doy), (id, d)) for id, d in data.keys() if str(id[:-2]) == str(locid))[1]
    return x



####### STEP 1: make this mapping from the MODIS data: 
#                  { year: {(locID, dayOfYear): pixel} }  

MODIS_DATA = {}
i=0
for dir in os.listdir(gee_root):
    if re.match('pixel_data_20\d\d', dir):
        year = int(re.findall('20\d\d', dir)[0])
        MODIS_DATA[year] = {}
        
        f = open(gee_root + dir + '/pixel.csv')
        a = {x: i for (i, x) in enumerate(next(f).strip().split(','))}   # get mapping from attribute name to index

        for line in f:
            line = line.strip().split(',')
            MODIS_DATA[year][(line[a['Location ID']], line[a['DayOfYear']])] = [int(x) for x in line[a['SolarZenith']: a['sur_refl_b07'] + 1]]


####### STEP 2: step through each label and pull out its corresponding pixel. print to csv at the end

s = 'diseased, loc_id, longitude, latitude, date, solar_zenith, state_qa, view_zenith, sr_b01, sr_b02, sr_b03, sr_b04, sr_b05, sr_b06, sr_b07\n'

for row, (label, lon, lat, locid, date) in enumerate(zip(labels.Diseased, labels.Longitude, labels.Latitude, labels['Location ID'], labels.ObsDate)):
    dt = datetime.strptime(re.findall('\d\d?/\d\d?/\d\d', date)[0], '%x')

    # throw out records w missing location or 2009 (lots of missing data that year)
    if (lon < -900) or (lat < -900) or (dt.year == 2009) or (label not in [0, 1]):
        continue

    key = get_closest_key(MODIS_DATA[dt.year], locid, get_DOY(dt))
    # lon and lat are strings so that i don't have to mess with float formatting
    s += '%d, %d, %s, %s, %s, %s\n' % \
        (label, locid, str(lon), str(lat), date, ', '.join(str(x) for x in MODIS_DATA[dt.year][key]))


print s
