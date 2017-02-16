"""
contains files for msc operations over geographical regions

=== USAGE
python region_utils.py ../../data/regions.kml ../../data/raw_survey.csv

=== TODO
 - ratio weighting
 - obs freq thresholding/weighting
 - final per-season labels
"""
import re
import sys
from kml_utils import RegionMap
import pandas as pd
from datetime import datetime
from collections import defaultdict
import time


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
    def maxStemStripe3Leaf(sevs):
        return max(sevs[0], sevs[2], (3 if sevs[1] == 3 else 0))

    @staticmethod
    def stemStripe2(sevs):
        return 1 if  (sevs[0] >= 2 or sevs[2] >= 2) else 0


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


    def __str__(self):
        s =  'OBSERVATION:\n'
        s += '\t lat: %s\n'
        s += '\t lon: %s\n'
        s += '\t severities: %s\n'
        s += '\t label: %s\n'
        s = s % (self.lat, self.lon, str(self.severities), self.label)
        if self.region:
            s += '\t region: %s\n' % self.region
        return s



class SurveyFeaturizer:
    def __init__(self, regions_file, survey_file, thresholder=Thresholders.stemStripe2):
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


    def surveyed_regions(self, season):
        """ returns all surveyed regions for a season """
        return self.bucketed_observations[season].keys()

    def count_obs(self, region, season):
        """ counts num observations of each type for a region/season """
        observations = self.bucketed_observations[season][region]

        nPos = sum(o.label * self.__weight(o) for o in observations if o.label > 0)
        nNeg = sum(1 * self.__weight(o) for o in observations if o.label == 0)

        return nPos, nNeg

    def score_region(self, region, season):
        """ get score of region for season. higher score means more 
            likely to be binned as diseased. score is weighted average
            of label ratios
        """
        observations = self.bucketed_observations[season][region]
        return sum(o.label * self.__weight(o) for o in observations) * 1.0 / len(observations)


    def ratio_region(self, region, season)
        nPos, nNeg = self.count_obs(region, season)
        return nPos * 1.0 / (nNeg or 0.5)


    def label(self, region, season, ratio=False):
        """ label a region for a season. 1 means dieseasd, 0 means disease-free

            TODO make this smarter! plot distribution!
        """
        if ratio:
            return 1.0 if self.region_ratio(region, season) > 1.0 else 0

        return 1.0 if self.score_region(region, season) > 1.5 else 0



    def __weight(self, o):
        """ weights observations according to gaussian that's skewed towards the end of the season
            intuitively, observations closer to the end of the season should matter more

            d is the day of the season
        """
        d = o.day_of_season
        return min(1, (d / 30) * 0.15)
        

        return 1.0

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

    # test with manually verified locations
    assert sf.count_obs('TOLE', 2015) == (1.0, 0.0)    # there was 1 pos ob in this season
    quit()
    assert sf.count_obs('TOLE', 2011) == (0.0, 1.0)    # 1 neg ob

    assert sf.count_obs('AMBO_ZURIA', 2011) == (1.0, 1.0) # etc
    assert sf.count_obs('AMBO_ZURIA', 2013) == (0.0, 4.0)

    assert sf.count_obs("DUGDA", 2013) == (0.0, 6.0)
    assert sf.count_obs("DUGDA", 2007) == (0.0, 1.0)





