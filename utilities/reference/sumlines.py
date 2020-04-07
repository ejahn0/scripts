import numpy as np
import pdb

snap = input('enter the snapshot number as a string: ')

# numlines_ar = np.array([])
# for i in range(4):
# 	filename = 'halos_'+snap+'.'+str(i)+'.ascii'

# 	numlines_ar = np.append(numlines_ar, sum(1 for line in open(filename)))

# total = sum(numlines_ar)

# print numlines_ar
# print total

dir_dmo = '/home/ethan/data/catalogs/dmo/ascii/'

numlines = 0
for i in range(4):
	filename = dir_dmo+'halos_'+snap+'.'+str(i)+'.ascii'
	numlines += sum(1 for line in open(filename))

print numlines
















pdb.set_trace()