import os
hostname = os.uname()[1]

#-raptor--------------------------------------------------------------
if hostname=='master':
	catdir = '/mainvol/ejahn/fire/catalogs/' 
	datdir = '/mainvol/ejahn/data/plots/hdf5/'
	helpdir = '/home/ejahn003/scripts/utilities/helpers/'
	plotdir = '/home/ejahn003/plots/'
	refdir = '/home/ejahn003/scripts/utilities/reference/'
	snapdir = '/mainvol/ejahn/fire/snapshots/'
	fileno_dir = '/home/ejahn003/scripts/utilities/reference/fileno_redshift.txt'
	smuggledir = '/mainvol/ejahn/smuggle/output/live_halo_02.2020/'
	projdir = '/home/ejahn003/projections/'
	firedir = '/mainvol/ejahn/fire/'


#-stampede------------------------------------------------------------
elif 'stampede2.tacc.utexas.edu' in hostname:
	firedir = '/scratch/projects/xsede/GalaxiesOnFIRE/'
	datdir='/home1/05033/ejahn/data/plots/hdf5/'
	helpdir = '/home1/05033/scripts/utilities/helpers/'
	plotdir='/home1/05033/ejahn/plots/'
	fileno_dir = '/home1/05033/ejahn/scripts/utilities/reference/fileno_redshift.txt'
	refdir = '/home1/05033/ejahn/scripts/utilities/reference/'


#-laptop--------------------------------------------------------------
elif hostname=='peregrin' or hostname=='eostrix':
	catdir = '/run/media/ethan/Watson/big_data/catalogs/'
	datdir = '/home/ethan/research/data/plots/hdf5/'
	helpdir = '/home/ethan/research/scripts/utilities/helpers/'
	plotdir = '/home/ethan/research/plots/'
	refdir = '/home/ethan/research/scripts/utilities/reference/'
	snapdir = '/run/media/ethan/Watson/big_data/snapshots/'
	fileno_dir = '/home/ethan/research/scripts/utilities/reference/fileno_redshift.txt'
	obsdir = '/home/ethan/research/data/observations/'
	firedir = '/run/media/ethan/Extreme SSD/research/data/fire/'


#-office-desktop------------------------------------------------------
elif hostname=='anattmux':
	datdir = '/home/ethan/research/data/plots/hdf5/'
	catdir = '/home/ethan/big_data/catalogs/md/'
	plotdir = '/home/ethan/research/plots/'
	m12dir = '/home/ethan/research/data/m12i/'
	dmodir = '/home/ethan/research/data/catalogs/dmo/m11q/hdf5/'
	refdir = '/home/ethan/research/scripts/reference/'
	snapdir = '/home/ethan/big_data/snapshots/'

#---------------------------------------------------------------------
else:
	raise ValueError('unknown host! :(')

