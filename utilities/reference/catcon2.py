import os
import h5py
import numpy as np 
import re
import pdb

dir_asc = '/home/ethan/research/data/catalogs/graus/'
dir_hdf = '/home/ethan/research/data/catalogs/graus/hdf5/'

# get a list of files in the current directory
# ideally this works best if the current directory only contains the files you wish to convert
allfiles = np.array(sorted(os.listdir(dir_asc)))

files_to_convert = np.array([])

for file in allfiles:
	if '.ascii' in file:
		files_to_convert = np.append(files_to_convert,file)

files_to_convert = files_to_convert[1:]


# loop through all the files in the given directory 
for name in files_to_convert:

	print 'reading ascii file '+name

	# open first file and get all the fields in the header
	# this will be used to create data tables in the new hdf5 file
	filename = dir_asc+name
	file = open(filename,'r')
	fields = np.array(file.readline().split(' '))
	fields[0] = re.sub(r'[^\w]','',fields[0])
	
	# loop through header
	header = np.array([])
	line = file.readline()
	while '#' in line:
		header = np.append(header, line)
		line = file.readline()

	# initialize data array
	data = np.array([])
	data = np.append( data, np.float64(line.split(' ')) )

	# loop through remaining lines
	for line in file:
		data = np.vstack(( data, np.float64(line.split(' ')) ))

	file.close()

	print 'finished reading ascii file, now writing hdf5 file'

	# sort by the virial mass from most to least massive, keeping rows intact
	# viewstring = ''
	# for i in range(len(fields)):
	# 	viewstring = viewstring + 'i8,'
	# viewstring = viewstring[:-1]
	viewstring = ('i8,'*len(fields))[:-1]
	data = np.sort(data.view(viewstring), order=['f2'], axis=0).view(np.float64)[::-1]

	# create and write the hdf5 file
	newfile = h5py.File(dir_hdf+name[0:-6]+'.hdf5','w')

	for i in range(len(fields)):
		newfile.create_dataset(fields[i],data=data[:,i])

	newfile.create_dataset('header',data=header)

	newfile.close()


pdb.set_trace()