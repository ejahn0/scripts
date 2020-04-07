import sys
import os


thisfile = sys.argv[0]
thisfile2 = os.path.basename(__file__)


print 'sys.argv[0]: ', thisfile
print 'os.path.basename(__file__): ', thisfile2


print '\n this is a file that does a thing' 