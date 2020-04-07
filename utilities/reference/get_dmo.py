import numpy as np 
import pdb

dir = '/home/ethan/data/catalogs/dmo/ascii/'
h 		= 0.702000
snap = input('enter snapshot number: ')

#create arrays for ALL data points
vmax_array = np.array([])
mvir_array = np.array([])


#loop through file parts 0-3
for i in range(4):

	#open the file
	# filename = dir+'halos_'+str(snap)+'.'+str(i)+'.ascii'
	filename = dir+'halos_'+str(snap)+'.'+str(i)+'.ascii'
	f = open(filename,'r')

	#loop through header
	line = f.readline()
	while '#' in line:
		line = f.readline()

	#now 'line' is the first line of the file with actual data
	#use current line to initialize 
	linestuff = np.array(line.split(' '))
	mvir = np.float64(linestuff[26])/(h**2)
	vmax = np.float64(linestuff[5])
	x = np.float64(linestuff[8])/(h**2)
	y = np.float64(linestuff[9])/(h**2)
	z = np.float64(linestuff[10])/(h**2)

	newline = str(vmax)+' '+str(mvir)+' '+str(x)+' '+str(y)+' '+str(z)+'\n'

	line_array = np.array([])
	line_array = np.append(line_array,newline)

	#loop through remaining data
	for line in f:
		linestuff = np.array(line.split(' '))
		vmax = np.float64(linestuff[5])
		mvir = np.float64(linestuff[26])/(h**2)
		x = np.float64(linestuff[8])/(h**2)
		y = np.float64(linestuff[9])/(h**2)
		z = np.float64(linestuff[10])/(h**2)
		newline = str(vmax)+' '+str(mvir)+' '+str(x)+' '+str(y)+' '+str(z)+'\n'
		line_array = np.append(line_array,newline)

	#file has been looped through completely at this point
	#close it and move to the next
	f.close()

#now the arrays should have data from all the files
#gotta write it

newfilename = '/home/ethan/Desktop/data_NEWcat_600-3.txt'
newfile = open(newfilename,'w')

for i in range(len(line_array)):
	newfile.write(line_array[i])

newfile.close()


pdb.set_trace()