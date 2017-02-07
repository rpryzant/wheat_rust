"""
Short script to change the "names" of each region in a kml to uppercase underscore.
Doing this because caps/underscores is my internal representation of region names


=== USAGE:
python correct_kml_names.py ../../data/regions.kml > out.kml
"""

import sys
import re

f = open(sys.argv[1]).read()


def repl(m):
    return m.groups()[0] + m.groups()[1].upper().replace(' ', '_') + m.groups()[2]

print re.sub('(<name>)(.*?)(</name>)', repl, f)







