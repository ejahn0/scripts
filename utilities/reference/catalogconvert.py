import os
import h5py
import numpy as np 
import re
import pdb

dir_asc = '/home/ethan/data/catalogs/dmo/ascii/'
dir_hdf = '/home/ethan/data/catalogs/dmo/hdf5/'

# get a list of files in the current directory
# ideally this works best if the current directory only contains the files you wish to convert
files = sorted(os.listdir(dir_asc))

nums = range(601)
for i in range(len(nums)):
	nums[i] = str(nums[i])
	if len(nums[i])==1:
		nums[i] = '00'+nums[i]
	elif len(nums[i])==2:
		nums[i] = '0'+nums[i]
	else:
		continue

files_to_convert = []
for n in nums:
	for f in files:
		if (n in f) & (not(n in files_to_convert)):
			files_to_convert.append(n)

for i in files_to_convert:
	print i

# loop through all the files in the given directory 
for num in files_to_convert:

	# open first file and get all the fields in the header
	# this will be used to create data tables in the new hdf5 file
	filename = dir_asc+'halos_'+num+'.0.ascii'
	firstfile = open(filename,'r')
	fields = np.array(firstfile.readline().split(' '))
	fields[0] = re.sub(r'[^\w]','',fields[0])
	
	# loop through header
	header = np.array([])
	line = firstfile.readline()
	while '#' in line:
		header = np.append(header, line)
		line = firstfile.readline()

	# initialize data array
	data = np.array([])
	data = np.append( data, np.float64(line.split(' ')) )

	# loop through remaining lines
	for line in firstfile:
		data = np.vstack(( data, np.float64(line.split(' ')) ))

	firstfile.close()

	print 'finished file 0'

	# loop through remaining 3 files with current designation to populate data array
	for i in [1,2,3]:
		filename = dir_asc+'halos_'+num+'.'+str(i)+'.ascii'
		f = open(filename,'r')

		# loop through header
		line = f.readline()
		while '#' in line:
			line = f.readline()

		# get current line
		data = np.vstack(( data, np.float64(line.split(' ')) ))

		# loop through remaining lines
		for line in f:
			data = np.vstack(( data, np.float64(line.split(' ')) ))

		f.close()

		print 'finished file',i

	# sort by the virial mass from most to least massive, keeping rows intact
	# viewstring = ''
	# for i in range(len(fields)):
	# 	viewstring = viewstring + 'i8,'
	# viewstring = viewstring[:-1]
	viewstring = ('i8,'*len(fields))[:-1]
	data = np.sort(data.view(viewstring), order=['f26'], axis=0).view(np.float64)[::-1]

	# create and write the hdf5 file
	newfile = h5py.File(dir_hdf+'halos_'+num+'.hdf5','w')

	for i in range(len(fields)):
		newfile.create_dataset(fields[i],data=data[:,i])

	newfile.create_dataset('header',data=header)

	newfile.close()

	print '------------------------------------'
	print 'created data set',num
	print '------------------------------------'


pdb.set_trace()