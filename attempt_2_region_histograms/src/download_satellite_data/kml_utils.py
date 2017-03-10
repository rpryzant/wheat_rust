"""
Classes, methods, and constants for wrangling kml files go here
=== USAGE
python kml_utils.py ../../data/regions.kml
"""
from pykml import parser
import sys
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon



class Region:
    """ class for managing particular regions.
    """
    def __init__(self, region_kml):
        self.name = region_kml.name
        s = str(region_kml.MultiGeometry.Polygon.outerBoundaryIs.LinearRing.coordinates)
        self.coords = [map(float, l.split(',')) for l in s.strip().split('\n')]
        self.polygon = Polygon(self.coords)

    def contains(self, lat, lon):
        """ does this region contain the point at (lat, lon)?
        """
        p = Point(lon, lat)
        return self.polygon.contains(p)



class RegionMap:
    """ Class that manages region map data
        Provides high-level interface for querying region maps
    """

    def __init__(self, kml_file):
        self.root = parser.fromstring(open(kml_file).read())
        self.regions = {}
        for region in self.root.Document.Placemark:
            # TODO WHAT DO IF MULTIPLE KEYS
            self.regions[self.__key(region.name)] = Region(region)

    def locate(self, lat, lon):
        """ look up region for given coordinate
        """
        for key, region in self.regions.iteritems():
            if region.contains(lat, lon):
                return key
        return None

    def __key(self, kml_name):
        """ prepares kml dump placemark name as a key 
            for internal mappings
        """
        return str(kml_name).upper().replace(' ', '_')



if __name__ == "__main__":
    rm = RegionMap(sys.argv[1])
    print '=== TESTING...'
    # should give city of Bahir Dar
    assert rm.locate(11.594345, 37.391027) == 'BAHIR_DAR'
    # should give nothing - isn't in ethiopia
    assert rm.locate(15, 15) == None
    print '\t passed!'

