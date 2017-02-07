"""
contains files for msc operations over geographical regions

=== USAGE
python region_utils.py ../data/regions.kml ../data/raw_survey.csv

"""
import re
import sys
from kml_utils import RegionMap
import pandas as pd
from datetime import datetime
from collections import defaultdict

class PDUtils:
    @staticmethod
    def ts(r):
        """ makes datetime timestamp from a survey observation
        """
        s = r.ObsDate
        # expects string like '11/12/12 0:00' 
        return datetime.strptime(re.findall('\d\d?/\d\d?/\d\d', s)[0], '%x')

    @staticmethod
    def iter_rows(df):
        """ iterates through rows of pandas df """
        for rowi in range(len(df)):
            yield df.iloc[rowi,:]


class Thresholders:
    """ different thresholding strategies to cast observations as binary for disease """
    @staticmethod
    def maxSev(sevs):
        return max(sevs)

    @staticmethod
    def stemStripe2(sevs):
        return 1 if  (sevs[0] > 2 or sevs[2] > 2) else 0


class Observation:
    def __init__(self, pd_row, thresholder, region_map=None):
        ts = PDUtils.ts(pd_row)
        self.ts = ts
        self.season = self.__season(self.ts)
        self.day_of_season = self.__day(self.ts, self.season)
        self.lat = pd_row.Latitude
        self.lon = pd_row.Longitude
        self.severities = pd_row['Severity'], pd_row['Severity.1'], pd_row['Severity.2']        # (stem, leaf, stripe)
        self.label = thresholder(self.severities)
        if region_map:
            self.region = region_map.locate(self.lat, self.lon)

    def __season(self, ts):
        """ gets the growing season for a timestamp. Wheat season in ethiopia is **loosely**
            June 1st (6/1) -> Mar 1st (3/1), so anything before 3/1 is counted as
            belonging to the previous year's season

            precondition: ts is in the growing season (upstream steps should have filtered out 
                          observations at suspicious times
        """
        month = ts.month
        year = ts.year
        if month < 3:
            return year - 1
        return year

    def __day(self, ts, season):
        """ gets day of the season of this observation (assuming season start = 6/1)
        """
        season_start = datetime(season, 6, 1)
        return (ts - season_start).days


class SurveyFeaturizer:
    def __init__(self, regions_file, survey_file, thresholder=Thresholders.maxSev):
        self.thresholder = thresholder
        self.observations = []
        self.region_map = RegionMap(regions_file)
        self.bucketed_observations = defaultdict(lambda: defaultdict(list))    # {season: {region: [observations] } }

        # parse survey data and build out mappings
        survey_df = pd.read_csv(survey_file)
        for row in PDUtils.iter_rows(survey_df):
            if self.__healthy(row):
                obs = Observation(row, self.thresholder, self.region_map)
                if obs.region is None:    # we want everything to live in a region
                    continue
                self.observations.append(obs)
                self.bucketed_observations[obs.season][obs.region].append(obs)



    def __healthy(self, row):
        """ tests whether a row should be discarded """
        ts = PDUtils.ts(row)
        if ( row.Latitude < 0 or row.Longitude < 0 ) or \
           ( ts.month > 3 and ts.month < 6 ) or \
           ( row['Severity'] < 0 or row['Severity.1'] < 0 or row['Severity.2'] < 0 ) or \
           ts.year == 2009:
            return False

        return True



if __name__ == "__main__":
    regions = sys.argv[1]
    surveys = sys.argv[2]

    sf = SurveyFeaturizer(regions, surveys, Thresholders.stemStripe2)












