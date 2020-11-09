import numpy as np 
import catalogHDF5 as cat
import directories as d 
import math_helper as m
import snapHDF5, h5py, sys, fit, time
from os import listdir
from os.path import isfile, join
h = 0.7

#---Fire----------------------------------------------------------------------------
def calculate_massdist(simname,num='600'):
	
	print('calculating ' + simname)
	a, mvir, mstr, d_halo, r50, radius, halo_pos, hostIndex = cat.read(simname, 0, 'snapshot:scalefactor',
		'mass.200m','star.mass','distance.1d','star.radius.50','star.radius.90','star.position','host.index')

	h, omega_m, omega_l, f_b = cat.read(simname,0,'cosmology:hubble','cosmology:omega_matter','cosmology:omega_lambda','cosmology:baryon.fraction')
	H0 = 100*h; H0G = H0*to_Gyr

	r50_host = r50[hostIndex]; mvir_host = mvir[hostIndex]; radius_host = radius[hostIndex]; host_pos = halo_pos[hostIndex]
	rvir = virialradius(mvir_host,simname)

	nloops = 50
	d = None
	dmin = np.log10(rvir*1.e-3)
	dmax = np.log10(rvir)
	drange = np.logspace(dmin,dmax,nloops)
	mvir 	 = mvir[1:]
	mstr 	 = mstr[1:]
	d_halo 	 = d_halo[1:]
	radius   = radius[1:]
	halo_pos = halo_pos[1:]
	halo_edge_in  = d_halo - radius
	halo_edge_out = d_halo + radius

	snapfile = snapdir + simname + '/snapshot_' + num 
	darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h 
	dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
	d_dark	 = np.linalg.norm(dark_pos-host_pos,axis=1)
	Mdark_all_spheres = np.array([])
	d_in = 0.0; Mstr_previous = 0.0; Mgas_previous = 0.0; Mdark_previous = 0.0; n_prev=0

	sys.stdout.write("[%s]" % (" " * (columns-2)))
	sys.stdout.flush()
	sys.stdout.write("\b" * (columns-2)) 
	printThing='-'
	for d_out in drange:
		dark_select		= ((d_dark >= d_in) & (d_dark < d_out))
		d_dark_shell	= d_dark[dark_select]
		darkmass_shell	= darkmass[dark_select]
		dark_pos_shell	= dark_pos[dark_select]

		Mdark_sphere = Mdark_previous + np.sum(darkmass_shell)
		Mdark_all_spheres = np.append(Mdark_all_spheres, Mdark_sphere)
				
		d_in = d_out

		Mdark_previous = Mdark_sphere
		
		i = np.float(np.where(drange==d_out)[0][0])
		n = np.int((columns-1)*(i/np.float(nloops)))/len(printThing)
		if not(n==n_prev):
			sys.stdout.write(printThing)
			sys.stdout.flush()
			n_prev = n
		else:
			continue
	sys.stdout.write("\n")

	newfile = h5py.File(datdir+'massprofle_'+simname[0:4]+'.hdf5','w')
	newfile.create_dataset('drange',data=drange)
	newfile.create_dataset('M_dark_all',data=Mdark_all_spheres)
	newfile.close()

def calculate_stellar_kinematics():
	fig, ax = makefig(n_panels=1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	h, omega_m, omega_l, f_b = cat.read('m11q_res880',0,'cosmology:hubble','cosmology:omega_matter','cosmology:omega_lambda',
		'cosmology:baryon.fraction')
	H0 = 100*h; H0G = H0*to_Gyr

	#----------------------------------------------------------------------------------------------
	hostIndex, mstr, mvir, dist, starIDs_RS, position, r200m, r90, velocity = cat.read('m11q_res880',0,'host.index','star.mass','mass.200m','distance.1d',
		'star.indices','position','radius','star.radius.90','velocity')
	rvir = r200m[hostIndex]

	#-read-snapshots-------------------------------------------------------------------------------
	snapfile = snapdir + 'm11q_res880' + '/snapshot_600'

	star_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h - position[hostIndex]
	star_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=4) - velocity[hostIndex]
	starIDs_SS = snapHDF5.read_block(snapfile, 'ID  ', parttype=4) 
	'''
	this_sat_radius = sat_radius[j]
	this_sat_position = sat_position[j]
	this_satIDs_cat = sat_strIDs_catalog[j]
	this_satIDs_cat = star_IDs[this_satIDs_cat].astype(int)

	stardist = np.linalg.norm(star_pos - this_sat_position, axis=1)

	select_stars_z0 = (stardist < this_sat_radius) #see diagram
	Age_z0_sat = star_ages[select_stars_z0]
	IDs_z0_sat = star_IDs[select_stars_z0].astype(int)
	pos_z0_sat = star_pos[select_stars_z0]
	'''

	#----------------------------------------------------------------------------------------------
	starIDs_RS_host = starIDs_RS[hostIndex]
	starIDs_SS_host = starIDs_SS[starIDs_RS_host]

	all_pos_mag = np.linalg.norm(star_pos,axis=1)
	select = (all_pos_mag < r90[hostIndex])
	starIDs_SS_man = starIDs_SS[select]

	compare = np.in1d(starIDs_SS_host,starIDs_SS_man)

	print(len(starIDs_SS_host))
	print(len(starIDs_SS_man))
	print(len(starIDs_SS_man[compare]))


	# #----------------------------------------------------------------------------------------------
	# pos = star_pos[starIDs_rockstar]
	# pos_mag = all_pos_mag[starIDs_rockstar]

	# rhat = np.divide(pos.T,pos_mag).T

	# vel = star_vel[starIDs_rockstar] 

	#----------------------------------------------------------------------------------------------

	# vel_rad = np.einsum('ij,ij->i',vel, rhat)

	# vel_tan = np.array([])

	# for i in range(len(vel)):
	# 	printthing = str(i+1)+'/'+str(len(vel))
	# 	sys.stdout.write(printthing)
	# 	sys.stdout.flush()
	# 	sys.stdout.write("\b" * (len(printthing)))
	# 	vel_tan = np.append(vel_tan, np.linalg.norm(np.cross(rhat[i],vel[i])))

	# print('len v.rad = ',len(vel_rad))
	# print('len v.tan = ',len(vel_tan))


	# f = h5py.File(datdir+'/stellar_velocities.hdf5','w')
	# f.create_dataset('v.rad',data=vel_rad)
	# f.create_dataset('v.tan',data=vel_tan)
	# f.close()

def calculate_gas_fraction():
	sims,labels,host_mvir,host_mstr,colors,rvirs,opcs,zords,linst,linst2,has_dmo,min_vmax = getsims('all')
	
	newfile = h5py.File(datdir+'m12_sat_gas_fractions.hdf5','w')

	for sim in m12sims:
		print(sim)
		i = np.where(m12sims==sim)[0][0]
		h, omega_m, omega_l, f_b = cat.read(sim,0,'cosmology:hubble','cosmology:omega_matter','cosmology:omega_lambda','cosmology:baryon.fraction')
		H0 = 100*h; H0G = H0*to_Gyr

		snapfile = snapdir + sim + '/snapshot_600'
		gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
		gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h

		hostIndex, mstr, r200m, m200, dist, Nstar, r90, str_position = cat.read(sim,0,'host.index','star.mass','radius','mass.200m',
			'distance.1d','star.number','star.radius.90','star.position')
		hostradius = r200m[hostIndex]
		sel = (dist > 0) & (dist < hostradius) & (mstr > 0)# & (Nstar > 100)
		sat_m200 = m200[sel]
		sat_mstr = mstr[sel]
		sat_pos = str_position[sel]
		sat_r90 = r90[sel]
		n_sats = np.count_nonzero(sel)

		for j in range(n_sats):
			print(str(j+1)+'/'+str(n_sats))
			sel_gas = (np.linalg.norm(gas_pos - sat_pos[j], axis=1) < sat_r90[j])
			this_gasmass = np.sum(gasmass[sel_gas])
			newfile.create_dataset(sim[0:4]+'_'+str(j),data=this_gasmass)

	newfile.close

def calculate_subhalo_profile_time_average(n_snaps=60,vcut=10):
	sims,labels,host_mvir,host_mstr,colors,rvirs,opcs,zords,linst,linst2,has_dmo,min_vmax = getsims('all')
	dcuts = np.array([1,0.4,0.2])

	newfile = h5py.File(datdir+'subhalo_profile_vcut'+str(vcut)+'_tavg.hdf5','w')

	snapnums = np.arange((600-n_snaps),601).astype('str')

	sims = np.append(sims,m12sims)

	for sim in sims:
		i = np.where(sims==sim)[0][0]
		h, omega_m, omega_l, f_b = cat.read(sim,0,'cosmology:hubble','cosmology:omega_matter','cosmology:omega_lambda',
			'cosmology:baryon.fraction')
		H0 = 100*h; H0G = H0*to_Gyr
		print(sim)
		# print ''

		for snap in snapnums:
			printthing = 'calculating snapshot '+snap+'/600'
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(np.where(snapnums==snap)[0][0]==len(snapnums)-1):
				sys.stdout.write("\b" * (len(printthing)))
			else:
				print('\n')

			j = np.where(snapnums==snap)[0][0]
			# print(type(snap))
			hostIndex, mass, dist, vmax, r200m = cat.read(sim,snap,'host.index','mass.200m','distance.1d','vel.circ.max','radius')
			radius = r200m[hostIndex]

			select = (dist > 0.) & (dist < radius) & (vmax > vcut)
			dist = np.sort(dist[select])
			num = np.arange(len(dist))

			dist_rs = np.logspace(np.log10(10), np.log10(300))


			#---resample-vmax-curve-and-save-------------------------------------------------------------------
			if np.amax(dist_rs) > np.amax(dist):
				dist = np.append(dist, np.amax(dist_rs)*1.01)
				num = np.append(num, 0)

			if np.amin(dist_rs) < np.amin(dist):
				distfunc = scint.interp1d(dist, num)
				num_rs = distfunc(dist_rs[dist_rs > np.amin(dist)])

				while(len(num_rs) < len(dist_rs)):
					num_rs = np.append(np.nan, num_rs)

			else:
				distfunc = scint.interp1d(dist, num)
				num_rs = distfunc(dist_rs)

			if j==0:
				num_rs_array = num_rs
			else:
				num_rs_array = np.vstack((num_rs_array, num_rs))



		#--------------------------------------------------------------------------------------------------
		num_avg = np.array([]) 
		p_error = np.array([]) 
		n_error = np.array([])
		
		for k in range(len(dist_rs)):
			column = num_rs_array[:,k]
			column = column[np.logical_not(np.isnan(column))]

			num_avg = np.append(num_avg, np.mean(column))

			p_error = np.append(p_error, np.mean(column) + np.std(column[column > np.mean(column)]))
			n_error = np.append(n_error, np.mean(column) - np.std(column[column < np.mean(column)]))

			# column_dmo = num_rs_array_dmo[:,k]
			# column_dmo = column_dmo[np.logical_not(np.isnan(column_dmo))]
			# num_dmo_avg = np.append(num_dmo_avg, np.mean(column_dmo))
			# p_error_dmo = np.append(p_error_dmo, np.mean(column_dmo) + np.std(column_dmo[column_dmo > np.mean(column_dmo)]))
			# n_error_dmo = np.append(n_error_dmo, np.mean(column_dmo) - np.std(column_dmo[column_dmo < np.mean(column_dmo)]))

		newfile.create_dataset(sim[0:4]+'.vcut'+str(vcut)+'.hydro',data=num_avg)
		newfile.create_dataset(sim[0:4]+'.vcut'+str(vcut)+'.hydro.pos.error',data=p_error)
		newfile.create_dataset(sim[0:4]+'.vcut'+str(vcut)+'.hydro.neg.error',data=n_error)
		# if has_dmo[i]:
		# newfile.create_dataset(sim[0:4]+'.vcut'+str(vcut)+'.dmo',data=num_dmo_avg)
		# newfile.create_dataset(sim[0:4]+'.vcut'+str(vcut)+'.dmo.pos.error',data=p_error_dmo)
		# newfile.create_dataset(sim[0:4]+'.vcut'+str(vcut)+'.dmo.neg.error',data=n_error_dmo)
	newfile.close()

def density_v_time():
	snapdir = '/mainvol/ejahn/smuggle/output/live_halo/SFhigh_1e5/'
	snaplist = np.arange(9,201)
	h = 0.7

	rho = np.array([])
	time = np.array([])

	for snapnum in snaplist:
		snapnum = str(snapnum).zfill(3)
		snapfile = snapdir+'/snapshot_'+snapnum
		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
		dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
		header = snapHDF5.snapshot_header(snapfile)
	

#---smuggle-------------------------------------------------------------------------
def calculate_CoM(sim):
	newfilename = d.datdir+'centerofmass_'+sim+'.hdf5'
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			CoM_all = np.array(newfile['CoM'])
			min_snap = len(CoM_all)
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim
	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if (min_snap)>max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		for snapnum in np.arange(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			darkpos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h

			x_cm = np.sum(darkpos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(darkpos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(darkpos[:,2] * darkmass) / np.sum(darkmass)
			cm = np.array([x_cm,y_cm,z_cm]).T

			# CoM_all = np.append(CoM_all,cm)

			if snapnum == 0:
				CoM_all = cm
			else:
				CoM_all = np.vstack((CoM_all,cm))


		if preexist:
			del newfile['CoM']

		newfile.create_dataset('CoM',data=CoM_all)
		print('\nwrote data to file')

	newfile.close()	
	print('---------------------------------------------------------------'+'\n')

def calculate_vel_CoM(sim):
	newfilename = d.datdir+'centerofmassvelocity_'+sim+'.hdf5'
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			v_cm_all = np.array(newfile['Vel_CoM'])
			min_snap = len(v_cm_all)
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim
	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if (min_snap)>max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		for snapnum in np.arange(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			dark_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=1)
			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h

			vx_cm = np.sum(dark_vel[:,0] * darkmass) / np.sum(darkmass)
			vy_cm = np.sum(dark_vel[:,1] * darkmass) / np.sum(darkmass)
			vz_cm = np.sum(dark_vel[:,2] * darkmass) / np.sum(darkmass)
			v_cm = np.array([vx_cm,vy_cm,vz_cm]).T

			# CoM_all = np.append(CoM_all,cm)

			if snapnum == 0:
				v_cm_all = v_cm
			else:
				v_cm_all = np.vstack((v_cm_all,v_cm))


		if preexist:
			del newfile['Vel_CoM']

		newfile.create_dataset('Vel_CoM',data=v_cm_all)
		print('\nwrote data to file')

	newfile.close()	
	print('---------------------------------------------------------------'+'\n')

def calculate_mass_in_r(sim):
	snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'
	h=0.7
	all_time = np.array([])

	print('analyzing simulation: '+sim+'\n\n------------------------------------------------\n\n')

	for n in np.arange(2,201):
		printthing = 'calculating snapshot '+str(n).zfill(3)+'/200'
		sys.stdout.write(printthing)
		sys.stdout.flush()
		sys.stdout.write("\b" * (len(printthing)))

		fname = d.smuggledir + sim + '/snapshot_' + str(n).zfill(3)
		header = snapHDF5.snapshot_header(fname)
		all_time = np.append(all_time, header.time)

		snapfile = snapdir+sim+'/snapshot_'+str(n).zfill(3)
		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
		dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

		gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
		gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h

		#-calculate-center-of-mass---------------------------------------------
		x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
		dark_cm = np.array([x_cm,y_cm,z_cm]).T

		d_dark = np.linalg.norm(dark_pos-dark_cm, axis=1)
		d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)

		#-calculate-mass-in-different-radii------------------------------------
		mdark_in_01 = np.sum(darkmass[(d_dark < 0.1)])
		mdark_in_02 = np.sum(darkmass[(d_dark < 0.2)])
		mdark_in_05 = np.sum(darkmass[(d_dark < 0.5)])
		mdark_in_1 = np.sum(darkmass[(d_dark < 1)])
		mdark_in_2 = np.sum(darkmass[(d_dark < 2)])
		mdark_in_5 = np.sum(darkmass[(d_dark < 5)])

		mgas_in_01 = np.sum(gasmass[(d_gas < 0.1)])
		mgas_in_02 = np.sum(gasmass[(d_gas < 0.2)])
		mgas_in_05 = np.sum(gasmass[(d_gas < 0.5)])
		mgas_in_1 = np.sum(gasmass[(d_gas < 1)])
		mgas_in_2 = np.sum(gasmass[(d_gas < 2)])
		mgas_in_5 = np.sum(gasmass[(d_gas < 5)])

		if n==2:
			darkmasses_in_Rs = np.array([mdark_in_01,mdark_in_02,mdark_in_05,mdark_in_1,mdark_in_2,mdark_in_5])
			gasmasses_in_Rs = np.array([mgas_in_01,mgas_in_02,mgas_in_05,mgas_in_1,mgas_in_2,mgas_in_5])
		else:
			this_darkmasses_in_Rs = np.array([mdark_in_01,mdark_in_02,mdark_in_05,mdark_in_1,mdark_in_2,mdark_in_5])
			this_gasmasses_in_Rs = np.array([mgas_in_01,mgas_in_02,mgas_in_05,mgas_in_1,mgas_in_2,mgas_in_5])

			darkmasses_in_Rs = np.vstack((darkmasses_in_Rs,this_darkmasses_in_Rs))
			gasmasses_in_Rs = np.vstack((gasmasses_in_Rs,this_gasmasses_in_Rs))

	datfname = d.datdir+'mass_in_r_'+sim+'.hdf5'
	print('\nsaving file: '+datfname+'\n\n')
	newfile = h5py.File(datfname,'w')
	newfile.create_dataset('darkmasses_in_Rs',data=darkmasses_in_Rs)
	newfile.create_dataset('gasmasses_in_Rs',data=gasmasses_in_Rs)
	newfile.create_dataset('time',data=all_time)
	newfile.close()

def calculate_sigma(sim):
	# snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'

	h=0.7
	all_time = np.array([])
	# drange = np.logspace(-1,2,100)
	print('analyzing simulation: '+sim)

	sigmafile = h5py.File(d.datdir+'sigma_v_annuli'+savenames[i]+'.hdf5','w')

	# distances = np.array([0.1,0.2,0.5,1,2,5])
	distances = np.array([0.1,0.3,0.5,1,2,4])

	for n in np.arange(10,201):
		printthing = 'calculating snapshot '+str(n).zfill(3)+'/200'
		sys.stdout.write(printthing)
		sys.stdout.flush()
		if not(n==200):
			sys.stdout.write("\b" * (len(printthing)))

		#----------------------------------------------------------------------
		#---read-simulation-data-----------------------------------------------
		#----------------------------------------------------------------------
		snapfile = snapdir+sim+'/snapshot_'+str(n).zfill(3)
		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
		dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
		dark_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=1)

		starmass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
		star_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h
		star_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=4)

		gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
		gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
		gas_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=0)

		rho = snapHDF5.read_block(snapfile,"RHO ",parttype=0)
		rho *= m.Xh/m.PROTONMASS*m.UnitDensity_in_cgs  #now in cm^{-3} h^3 
		U = snapHDF5.read_block(snapfile,"U   ",parttype=0)
		Nelec = snapHDF5.read_block(snapfile,"NE  ",parttype=0)
		MeanWeight= 4.0/(3*m.Xh+1+4*m.Xh*Nelec) * m.PROTONMASS
		temp = MeanWeight/m.BOLTZMANN * (m.gamma-1) * U * m.UnitEnergy_in_cgs/ m.UnitMass_in_g

		#----------------------------------------------------------------------
		#---center-coordinates-on-CoM------------------------------------------
		#----------------------------------------------------------------------
		x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
		dark_cm = np.array([x_cm,y_cm,z_cm]).T

		vx_cm = np.sum(dark_vel[:,0] * darkmass) / np.sum(darkmass)
		vy_cm = np.sum(dark_vel[:,1] * darkmass) / np.sum(darkmass)
		vz_cm = np.sum(dark_vel[:,2] * darkmass) / np.sum(darkmass)
		dark_v_cm = np.array([vx_cm,vy_cm,vz_cm]).T

		d_star = np.linalg.norm(star_pos-dark_cm, axis=1)
		d_gas  = np.linalg.norm(gas_pos-dark_cm, axis=1)

		star_pos = star_pos-dark_cm
		gas_pos = gas_pos-dark_cm
		star_vel = star_vel-dark_v_cm
		gas_vel = gas_vel-dark_v_cm

		r_star = np.sqrt(star_pos[:,0]**2 + star_pos[:,1]**2)
		r_gas = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2)

		for i in range(len(distances)-1):
			#-----------------------------------------------------------
			#---calculate-sigma-for-stars-------------------------------
			#-----------------------------------------------------------
			sel_stars_in_d = (r_star > distances[i]) & (r_star < distances[i+1])
			Nstar = np.count_nonzero(sel_stars_in_d)
			if Nstar < 2:
				sigma_rho = 0
				sigma_phi = 0
				sigma_z   = 0
			else:
				star_vel_in_d = star_vel[sel_stars_in_d]
				star_pos_in_d = star_pos[sel_stars_in_d]
				
				x = star_pos_in_d[:,0]
				y = star_pos_in_d[:,1]
				z = star_pos_in_d[:,2]
				vx = star_vel_in_d[:,0]
				vy = star_vel_in_d[:,1]
				vz = star_vel_in_d[:,2]

				rho = np.sqrt(x**2 + y**2)
				phi = np.arctan(x/y)

				vrho = (x*vx + y*vy)*(np.cos(phi) + np.sin(phi))/rho
				vphi = (x*vy - y*vx)*(np.cos(phi) - np.sin(phi))/rho

				vbar_rho = np.mean(vrho)
				vbar_phi = np.mean(vphi)
				vbar_z = np.mean(vz)

				sigma_rho   = np.sqrt(1./(Nstar-1.) * np.sum((vrho - vbar_rho)**2.))
				sigma_phi = np.sqrt(1./(Nstar-1.) * np.sum((vphi - vbar_phi)**2.))
				sigma_z   = np.sqrt(1./(Nstar-1.) * np.sum((vz - vbar_z)**2.))

			if i == 0:
				sigma_star = np.array([sigma_rho,sigma_phi,sigma_z])
			else:
				sigma_star = np.vstack((sigma_star,np.array([sigma_rho,sigma_phi,sigma_z])))

			#-----------------------------------------------------------
			#---calculate-sigma-for-gas---------------------------------
			#-----------------------------------------------------------
			sel_gas_in_d = (r_gas > distances[i]) & (r_gas < distances[i+1])
			Ngas = np.count_nonzero(sel_gas_in_d)
			
			sel_cold = (r_gas > distances[i]) & (r_gas < distances[i+1]) & (temp < 2e3)
			Ncold = np.count_nonzero(sel_cold)

			#---calculate-sigma-for-all-gas----------------------------
			if Ngas < 2:
				sigma_rho = 0
				sigma_phi = 0
				sigma_z   = 0
			else:
				gas_vel_in_d = gas_vel[sel_gas_in_d]
				gas_pos_in_d = gas_pos[sel_gas_in_d]
				x = gas_pos_in_d[:,0]
				y = gas_pos_in_d[:,1]
				z = gas_pos_in_d[:,2]
				vx = gas_vel_in_d[:,0]
				vy = gas_vel_in_d[:,1]
				vz = gas_vel_in_d[:,2]

				rho = np.sqrt(x**2 + y**2)
				phi = np.arctan(x/y)

				vrho = (x*vx + y*vy)*(np.cos(phi) + np.sin(phi))/rho
				vphi = (x*vy - y*vx)*(np.cos(phi) - np.sin(phi))/rho

				vbar_rho = np.mean(vrho)
				vbar_phi = np.mean(vphi)
				vbar_z = np.mean(vz)

				sigma_rho   = np.sqrt(1./(Ngas-1.) * np.sum((vrho - vbar_rho)**2.))
				sigma_phi = np.sqrt(1./(Ngas-1.) * np.sum((vphi - vbar_phi)**2.))
				sigma_z   = np.sqrt(1./(Ngas-1.) * np.sum((vz - vbar_z)**2.))

			#---calculate-sigma-for-cold-gas---------------------------
			if Ncold < 2:
				sigma_rho_cold = 0
				sigma_phi_cold = 0
				sigma_z_cold   = 0
			else:
				gas_vel_in_d = gas_vel[sel_cold]
				gas_pos_in_d = gas_pos[sel_cold]
				x = gas_pos_in_d[:,0]
				y = gas_pos_in_d[:,1]
				z = gas_pos_in_d[:,2]
				vx = gas_vel_in_d[:,0]
				vy = gas_vel_in_d[:,1]
				vz = gas_vel_in_d[:,2]

				rho = np.sqrt(x**2 + y**2)
				phi = np.arctan(x/y)

				vrho = (x*vx + y*vy)*(np.cos(phi) + np.sin(phi))/rho
				vphi = (x*vy - y*vx)*(np.cos(phi) - np.sin(phi))/rho

				vbar_rho = np.mean(vrho)
				vbar_phi = np.mean(vphi)
				vbar_z = np.mean(vz)

				sigma_rho_cold   = np.sqrt(1./(Ncold-1.) * np.sum((vrho - vbar_rho)**2.))
				sigma_phi_cold = np.sqrt(1./(Ncold-1.) * np.sum((vphi - vbar_phi)**2.))
				sigma_z_cold   = np.sqrt(1./(Ncold-1.) * np.sum((vz - vbar_z)**2.))


			#---add-data-to-array------------------------------------------
			if i == 0:
				sigma_gas = np.array([sigma_rho,sigma_phi,sigma_z])
				sigma_coldgas = np.array([sigma_rho_cold,sigma_phi_cold,sigma_z_cold])
			else:
				sigma_gas = np.vstack((sigma_gas,np.array([sigma_rho,sigma_phi,sigma_z])))
				sigma_coldgas = np.vstack((sigma_coldgas,np.array([sigma_rho_cold,sigma_phi_cold,sigma_z_cold])))

		#---------------------------------------------------------------
		#---------------------------------------------------------------
		sigmafile.create_dataset('sigma_star_'+str(n).zfill(3),data=sigma_star)
		sigmafile.create_dataset('sigma_gas_'+str(n).zfill(3),data=sigma_gas)
		sigmafile.create_dataset('sigma_coldgas_'+str(n).zfill(3),data=sigma_coldgas)
	
	print('\nwriting file: '+d.datdir+'sigma_v_annuli'+sim+'.hdf5')
	sigmafile.close()

	print('---------------------------------------------------------')

def calculate_timefile():
	
	timefile = h5py.File(d.datdir+'timefile.hdf5','w')

	for sim in models:
		all_time = np.array([])
		print(sim+'\n')

		for n in np.arange(1,201):
			printthing = 'calculating snapshot '+str(n).zfill(3)+'/200'
			sys.stdout.write(printthing)
			sys.stdout.flush()
			sys.stdout.write("\b" * (len(printthing)))

			fname = d.smuggledir + sim + '/snapshot_' + str(n).zfill(3)
			header = snapHDF5.snapshot_header(fname)
			all_time = np.append(all_time, header.time)

		print('\n-----------------------------------------\n')
		timefile.create_dataset('alltime_'+sim,data=all_time)

	timefile.close()
		 
def calculate_sigma_profiles(sim):#,snapnum=400):
	# snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'

	# if not(sim in models):
	# 	raise ValueError('please choose a sim in models')

	# j = np.where(models==sim)[0][0]

	h=0.7
	all_time = np.array([])
	# drange = np.logspace(-1,2,100)
	print('analyzing simulation: '+sim)

	sigmafile = d.datdir+'sigma_profiles_'+sim+'.hdf5'

	# # sigmafile = h5py.File(sigmaname,'w')

	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(sigmafile))
	drange = np.logspace(-1,1,100)

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(sigmafile,'r+')

		try:
			# sigma_star   = np.array(newfile['sigma_star'])
			# sigma_gas  = np.array(newfile['sigma_gas'])
			# sigma_coldgas = np.array(newfile['sigma_coldgas'])
			# min_snap = sigma_star.shape[0]
			keys = np.array(newfile.keys()).astype('str')
			min_snap = int(keys[-1][-3:])+1
			
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 10
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(sigmafile,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join
	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()

		v_cm_file = h5py.File(d.datdir+'centerofmassvelocity_'+sim+'.hdf5','r')
		v_cm_all = np.array(v_cm_file['Vel_CoM'])
		v_cm_file.close()

		for snapnum in range(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))
			#----------------------------------------------------------------------------------------------
			#---read-simulation-data-----------------------------------------------------------------------
			#----------------------------------------------------------------------------------------------
			# snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
			dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
			dark_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=1)

			starmass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
			star_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h
			star_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=4)

			gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
			gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
			gas_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=0)

			rho = snapHDF5.read_block(snapfile,"RHO ",parttype=0)
			rho *= m.Xh/m.PROTONMASS*m.UnitDensity_in_cgs  #now in cm^{-3} h^3 
			U = snapHDF5.read_block(snapfile,"U   ",parttype=0)
			Nelec = snapHDF5.read_block(snapfile,"NE  ",parttype=0)
			MeanWeight= 4.0/(3*m.Xh+1+4*m.Xh*Nelec) * m.PROTONMASS
			temp = MeanWeight/m.BOLTZMANN * (m.gamma-1) * U * m.UnitEnergy_in_cgs/ m.UnitMass_in_g

			#----------------------------------------------------------------------------------------------
			#---center-coordinates-on-CoM------------------------------------------------------------------
			#----------------------------------------------------------------------------------------------
			# x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
			# y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
			# z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
			# dark_cm = np.array([x_cm,y_cm,z_cm]).T
			dark_cm = CoM_all[snapnum]

			# vx_cm = np.sum(dark_vel[:,0] * darkmass) / np.sum(darkmass)
			# vy_cm = np.sum(dark_vel[:,1] * darkmass) / np.sum(darkmass)
			# vz_cm = np.sum(dark_vel[:,2] * darkmass) / np.sum(darkmass)
			# dark_v_cm = np.array([vx_cm,vy_cm,vz_cm]).T
			dark_v_cm = v_cm_all[snapnum]

			# d_star = np.linalg.norm(star_pos-dark_cm, axis=1)
			# d_gas  = np.linalg.norm(gas_pos-dark_cm, axis=1)

			star_pos = star_pos-dark_cm
			gas_pos = gas_pos-dark_cm
			star_vel = star_vel-dark_v_cm
			gas_vel = gas_vel-dark_v_cm

			# print(star_pos.shape)

			if len(star_pos) > 3:
				# print(star_pos.shape)
				# print(gas_pos.shape)
				r_star = np.sqrt(star_pos[:,0]**2 + star_pos[:,1]**2)
				r_gas = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2)

				#----------------------------------------------------------------------------------------------
				#---loop-through-distances-and-calculate-sigma-------------------------------------------------
				#----------------------------------------------------------------------------------------------
				distances = np.append(0.,np.logspace(-1,0.7,30))

				for i in np.arange(len(distances)-1):
					#---calculate-sigma-for-stars--------------------------------------------------------------
					sel_stars_in_d = (r_star > distances[i]) & (r_star < distances[i+1])
					Nstar = np.count_nonzero(sel_stars_in_d)
					if Nstar < 2:
						sigma_rho = 0
						sigma_phi = 0
						sigma_z   = 0
					else:
						star_vel_in_d = star_vel[sel_stars_in_d]
						star_pos_in_d = star_pos[sel_stars_in_d]
						
						x = star_pos_in_d[:,0]
						y = star_pos_in_d[:,1]
						z = star_pos_in_d[:,2]
						vx = star_vel_in_d[:,0]
						vy = star_vel_in_d[:,1]
						vz = star_vel_in_d[:,2]

						rho = np.sqrt(x**2 + y**2)
						phi = np.arctan(x/y)

						vrho = (x*vx + y*vy)*(np.cos(phi) + np.sin(phi))/rho
						vphi = (x*vy - y*vx)*(np.cos(phi) - np.sin(phi))/rho

						vbar_rho = np.mean(vrho)
						vbar_phi = np.mean(vphi)
						vbar_z = np.mean(vz)

						sigma_rho   = np.sqrt(1./(Nstar-1.) * np.sum((vrho - vbar_rho)**2.))
						sigma_phi = np.sqrt(1./(Nstar-1.) * np.sum((vphi - vbar_phi)**2.))
						sigma_z   = np.sqrt(1./(Nstar-1.) * np.sum((vz - vbar_z)**2.))

					if i == 0:
						sigma_star = np.array([sigma_rho,sigma_phi,sigma_z])
					else:
						sigma_star = np.vstack((sigma_star,np.array([sigma_rho,sigma_phi,sigma_z])))

					#---calculate-sigma-for-gas----------------------------------------------------------------
					sel_gas_in_d = (r_gas > distances[i]) & (r_gas < distances[i+1])
					Ngas = np.count_nonzero(sel_gas_in_d)

					sel_cold = (r_gas > distances[i]) & (r_gas < distances[i+1]) & (temp < 2e3)
					Ncold = np.count_nonzero(sel_cold)
					
					#---calculate-sigma-for-all-gas----------------------------
					if Ngas < 2:
						sigma_rho = 0
						sigma_phi = 0
						sigma_z   = 0
					else:
						gas_vel_in_d = gas_vel[sel_gas_in_d]
						gas_pos_in_d = gas_pos[sel_gas_in_d]
						x = gas_pos_in_d[:,0]
						y = gas_pos_in_d[:,1]
						z = gas_pos_in_d[:,2]
						vx = gas_vel_in_d[:,0]
						vy = gas_vel_in_d[:,1]
						vz = gas_vel_in_d[:,2]

						rho = np.sqrt(x**2 + y**2)
						phi = np.arctan(x/y)

						vrho = (x*vx + y*vy)*(np.cos(phi) + np.sin(phi))/rho
						vphi = (x*vy - y*vx)*(np.cos(phi) - np.sin(phi))/rho

						vbar_rho = np.mean(vrho)
						vbar_phi = np.mean(vphi)
						vbar_z = np.mean(vz)

						sigma_rho   = np.sqrt(1./(Ngas-1.) * np.sum((vrho - vbar_rho)**2.))
						sigma_phi = np.sqrt(1./(Ngas-1.) * np.sum((vphi - vbar_phi)**2.))
						sigma_z   = np.sqrt(1./(Ngas-1.) * np.sum((vz - vbar_z)**2.))

					#---calculate-sigma-for-cold-gas---------------------------
					if Ncold < 2:
						sigma_rho_cold = 0
						sigma_phi_cold = 0
						sigma_z_cold   = 0
					else:
						gas_vel_in_d = gas_vel[sel_cold]
						gas_pos_in_d = gas_pos[sel_cold]
						x = gas_pos_in_d[:,0]
						y = gas_pos_in_d[:,1]
						z = gas_pos_in_d[:,2]
						vx = gas_vel_in_d[:,0]
						vy = gas_vel_in_d[:,1]
						vz = gas_vel_in_d[:,2]

						rho = np.sqrt(x**2 + y**2)
						phi = np.arctan(x/y)

						vrho = (x*vx + y*vy)*(np.cos(phi) + np.sin(phi))/rho
						vphi = (x*vy - y*vx)*(np.cos(phi) - np.sin(phi))/rho

						vbar_rho = np.mean(vrho)
						vbar_phi = np.mean(vphi)
						vbar_z = np.mean(vz)

						sigma_rho_cold   = np.sqrt(1./(Ncold-1.) * np.sum((vrho - vbar_rho)**2.))
						sigma_phi_cold = np.sqrt(1./(Ncold-1.) * np.sum((vphi - vbar_phi)**2.))
						sigma_z_cold   = np.sqrt(1./(Ncold-1.) * np.sum((vz - vbar_z)**2.))


					#---add-data-to-array------------------------------------------
					if i == 0:
						sigma_gas = np.array([sigma_rho,sigma_phi,sigma_z])
						sigma_coldgas = np.array([sigma_rho_cold,sigma_phi_cold,sigma_z_cold])
					else:
						sigma_gas = np.vstack((sigma_gas,np.array([sigma_rho,sigma_phi,sigma_z])))
						sigma_coldgas = np.vstack((sigma_coldgas,np.array([sigma_rho_cold,sigma_phi_cold,sigma_z_cold])))
			
				#---write-calculated-sigmas-to-file------------------------------------------------------------
				newfile.create_dataset('sigma_star_'+str(snapnum).zfill(3),data=sigma_star)
				newfile.create_dataset('sigma_gas_'+str(snapnum).zfill(3),data=sigma_gas)
				newfile.create_dataset('sigma_coldgas_'+str(snapnum).zfill(3),data=sigma_coldgas)
			else:
				print('not enough star particles')
	print('\nfinished')
	newfile.close()

	print('---------------------------------------------------------------')

def calculate_gas_density(sim):
	
	snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'
	h=0.7
	all_time = np.array([])
	# drange = np.logspace(-1,2,100)
	print('analyzing simulation: '+sim)

	fname = d.datdir+'rhogas_'+sim+'.hdf5'
	gasfile = h5py.File(fname,'w')

	distances = np.array([0.1,0.2,0.5,1,2,5])
	
	all_time = np.array([])
	
	for snapnum in range(1,201):

		printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/200'
		sys.stdout.write(printthing)
		sys.stdout.flush()
		sys.stdout.write("\b" * (len(printthing)))

		#----------------------------------------------------------------------------------------------
		#---read-simulation-data-----------------------------------------------------------------------
		#----------------------------------------------------------------------------------------------
		snapfile = snapdir+sim+'/snapshot_'+str(snapnum).zfill(3)
		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
		dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
		dark_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=1)

		starmass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
		star_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h
		star_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=4)

		gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
		gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
		gas_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=0)

		header = snapHDF5.snapshot_header(snapfile)
		all_time = np.append(all_time, header.time)

		rho = snapHDF5.read_block(snapfile,"RHO ",parttype=0)
		rho *= m.Xh/m.PROTONMASS*m.UnitDensity_in_cgs  #now in cm^{-3} h^3 
		U = snapHDF5.read_block(snapfile,"U   ",parttype=0)
		Nelec = snapHDF5.read_block(snapfile,"NE  ",parttype=0)
		MeanWeight= 4.0/(3*m.Xh+1+4*m.Xh*Nelec) * m.PROTONMASS
		temp = MeanWeight/m.BOLTZMANN * (m.gamma-1) * U * m.UnitEnergy_in_cgs/ m.UnitMass_in_g

		#----------------------------------------------------------------------------------------------
		#---center-coordinates-on-CoM------------------------------------------------------------------
		#----------------------------------------------------------------------------------------------
		x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
		dark_cm = np.array([x_cm,y_cm,z_cm]).T

		d_gas  = np.linalg.norm(gas_pos-dark_cm, axis=1)

		gas_pos = gas_pos-dark_cm
		# r_gas = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2)

		rhogas = np.array([])
		rhogas_cold = np.array([])

		for dist in distances:
			gasmass_in_d = np.sum(gasmass[(d_gas < dist)])
			vol = (4./3.)*np.pi*(dist**3.)
			rhogas_in_d = gasmass_in_d/vol
			rhogas = np.append(rhogas,rhogas_in_d)

			gasmass_in_d_cold = np.sum(gasmass[(d_gas < dist) & (temp < 2.e3)])
			rhogas_in_d_cold = gasmass_in_d_cold/vol
			rhogas_cold = np.append(rhogas_cold,rhogas_in_d_cold)

		gasfile.create_dataset('snap_'+str(snapnum).zfill(3),data=rhogas)
		gasfile.create_dataset('cold_snap_'+str(snapnum).zfill(3),data=rhogas_cold)

	gasfile.create_dataset('all_time',data=all_time)
	gasfile.close()
	print('writing file: '+fname)
	print('------------------------------------------------')

def calculate_gas_density_from_snap(sim):
	# snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'
	h=0.7
	all_time = np.array([])
	# drange = np.logspace(-1,2,100)
	print('analyzing simulation: '+sim)

	if not(sim in models):
		raise ValueError('please choose an available sim')

	i = np.where(models==sim)[0][0]

	fname = d.datdir+'rhogas_fromsnap_'+savenames[i]+'.hdf5'
	gasfile = h5py.File(fname,'w')

	distances = np.array([0.1,0.2,0.5,1,2,5])
	
	all_time = np.array([])
	
	for snapnum in range(1,401):

		printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/400'
		sys.stdout.write(printthing)
		sys.stdout.flush()
		sys.stdout.write("\b" * (len(printthing)))

		#----------------------------------------------------------------------------------------------
		#---read-simulation-data-----------------------------------------------------------------------
		#----------------------------------------------------------------------------------------------
		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
		dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

		# gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
		gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h

		header = snapHDF5.snapshot_header(snapfile)
		all_time = np.append(all_time, header.time)

		rho = snapHDF5.read_block(snapfile,"RHO ",parttype=0)
		rho *= m.Xh/m.PROTONMASS*m.UnitDensity_in_cgs  #now in cm^{-3} h^3 
		U = snapHDF5.read_block(snapfile,"U   ",parttype=0)
		Nelec = snapHDF5.read_block(snapfile,"NE  ",parttype=0)
		MeanWeight= 4.0/(3*m.Xh+1+4*m.Xh*Nelec) * m.PROTONMASS
		temp = MeanWeight/m.BOLTZMANN * (m.gamma-1) * U * m.UnitEnergy_in_cgs/ m.UnitMass_in_g

		#----------------------------------------------------------------------------------------------
		#---center-coordinates-on-CoM------------------------------------------------------------------
		#----------------------------------------------------------------------------------------------
		x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
		dark_cm = np.array([x_cm,y_cm,z_cm]).T

		d_gas  = np.linalg.norm(gas_pos-dark_cm, axis=1)

		gas_pos = gas_pos-dark_cm
		# r_gas = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2)

		rhogas = np.array([])
		rhogas_cold = np.array([])

		for dist in distances:
			sel = (d_gas < dist)
			rhogas_in_d = np.mean(rho[sel])
			rhogas = np.append(rhogas,rhogas_in_d)

			sel_cold = (d_gas < dist) & (temp < 2.e3)
			rhogas_in_d_cold = np.mean(rho[sel_cold])
			rhogas_cold = np.append(rhogas_cold,rhogas_in_d_cold)

		gasfile.create_dataset('snap_'+str(snapnum).zfill(3),data=rhogas)
		gasfile.create_dataset('cold_snap_'+str(snapnum).zfill(3),data=rhogas_cold)

	gasfile.create_dataset('all_time',data=all_time)
	gasfile.close()
	print('writing file: '+fname)
	print('------------------------------------------------')

def print_total_mass(snapnum=200):
	snapdir = '/mainvol/ejahn/smuggle/output/live_halo_02.2020/fiducial_1e5/'
	snapfile = snapdir+'/snapshot_'+str(snapnum).zfill(3)
	
	darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
	# dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

	print(m.scinote(np.sum(darkmass)))

def calculate_mass_profiles_one_shot(sim,hires=True):
	if hires:
		newfilename = d.datdir+'massprofiles_'+sim+'_hires.hdf5'
		drange = np.logspace(-2,2,2000)
	else:
		newfilename = d.datdir+'massprofiles_'+sim+'.hdf5'
		drange = np.logspace(-1,np.log10(250),100)

	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	# drange = np.logspace(-2,np.log10(200),1000)
	# np.logspace(-2,np.log10(50),1000)

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			gas_profile_all   = np.array(newfile['gas'])
			dark_profile_all  = np.array(newfile['dark'])
			type2_profile_all = np.array(newfile['type2'])
			type3_profile_all = np.array(newfile['type3'])
			type4_profile_all = np.array(newfile['type4'])
			min_snap = gas_profile_all.shape[0]
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')
		dummy=''

	else:
		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		for snapnum in range(min_snap,max_snap+1):
			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))

			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
			dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

			x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
			dark_cm = np.array([x_cm,y_cm,z_cm]).T
			d_dark = np.linalg.norm(dark_pos-dark_cm, axis=1)

			gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
			gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
			d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)

			type2mass = snapHDF5.read_block(snapfile, 'MASS', parttype=2)*(1.e10)/h
			type2_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=2)/h
			d_type2 = np.linalg.norm(type2_pos-dark_cm, axis=1)

			type3mass = snapHDF5.read_block(snapfile, 'MASS', parttype=3)*(1.e10)/h
			type3_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=3)/h
			d_type3 = np.linalg.norm(type3_pos-dark_cm, axis=1)

			type4mass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
			type4_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h

			try:
				d_type4 = np.linalg.norm(type4_pos-dark_cm, axis=1)
				hastype4 = True
			except:
				hastype4 = False

			dark_profile,gas_profile,type2_profile,type3_profile,type4_profile = (np.array([]),)*5

			for dist in drange:
				dark_profile  = np.append(dark_profile, np.sum(darkmass[(d_dark < dist)]))
				gas_profile   = np.append(gas_profile,  np.sum(gasmass[(d_gas < dist)]))
				type2_profile = np.append(type2_profile,np.sum(type2mass[(d_type2 < dist)]))
				type3_profile = np.append(type3_profile,np.sum(type3mass[(d_type3 < dist)]))

				if hastype4: type4_profile = np.append(type4_profile,np.sum(type4mass[(d_type4 < dist)]))
				else: 		 type4_profile = np.append(type4_profile,0.)

			try:
				gas_profile_all   = np.vstack((gas_profile_all,gas_profile))
				dark_profile_all  = np.vstack((dark_profile_all,dark_profile))
				type2_profile_all = np.vstack((type2_profile_all,type2_profile))
				type3_profile_all = np.vstack((type3_profile_all,type3_profile))
				type4_profile_all = np.vstack((type4_profile_all,type4_profile))
			except:
				gas_profile_all   = gas_profile
				dark_profile_all  = dark_profile
				type2_profile_all = type2_profile
				type3_profile_all = type3_profile
				type4_profile_all = type4_profile

			# if np.any(dark_profile==0):
			# 	n_z = np.count_nonzero(dark_profile==0)
			# 	id_z = np.where(dark_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in dark_profile! occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')

			# if np.any(gas_profile==0):
			# 	n_z = np.count_nonzero(gas_profile==0)
			# 	id_z = np.where(gas_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in gas_profile! occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')

			# if np.any(type2_profile==0):
			# 	n_z = np.count_nonzero(type2_profile==0)
			# 	id_z = np.where(type2_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in type2_profile! occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')

			# if np.any(type3_profile==0):
			# 	n_z = np.count_nonzero(type3_profile==0)
			# 	id_z = np.where(type3_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in type3_profile! occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')

			# if np.any(dark_profile==0) or np.any(gas_profile==0) or np.any(type2_profile==0) or np.any(type3_profile==0):
			# 	raise ValueError('fix those zeros!')

			# if np.any(type4_profile==0):
			# 	n_z = np.count_nonzero(type4_profile==0)
			# 	id_z = np.where(type4_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in type4_profile, but that\'s actually possible, so we will proceed. occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')
		#---write--------------------------------------------------------------------------------------
		if preexist:
			del newfile['gas']; del newfile['dark']; del newfile['type2']; del newfile['type3']; del newfile['type4']; del newfile['drange']

		newfile.create_dataset('drange',data=drange)
		newfile.create_dataset('gas',   data=gas_profile_all)
		newfile.create_dataset('dark',  data=dark_profile_all)
		newfile.create_dataset('type2', data=type2_profile_all)
		newfile.create_dataset('type3', data=type3_profile_all)
		newfile.create_dataset('type4', data=type4_profile_all)
		# print('\nfinished')

	newfile.close()

	
	# print('---------------------------------------------------------------'+'\n')

def calculate_mass_profiles(sim,hires=True):
	'''
	this one writes data after every snapshot so
	it isn't lost if the connection is dropped
	'''
	if hires:
		newfilename = d.datdir+'massprofiles_'+sim+'_hires.hdf5'
		drange = np.logspace(-2,2,2000)
	else:
		newfilename = d.datdir+'massprofiles_'+sim+'.hdf5'
		drange = np.logspace(-1,np.log10(250),100)

	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))
	print('writing file after every snapshot to preserve data integrity')

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r')

		try:
			gas_profile_all   = np.array(newfile['gas'])
			if len(gas_profile_all.shape) == 0:
				min_snap = 0
			elif len(gas_profile_all.shape) == 1:
				min_snap = 1
			else:
				min_snap = gas_profile_all.shape[0]
			print('file exists: opening')
			preexist = True
			newfile.close()

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		# newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')
		dummy=''

	else:
		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		for snapnum in np.arange(min_snap,max_snap+1):
			#---print-current-snapshot-------------------------------------------------------------
			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))

			#---read-file--------------------------------------------------------------------------
			if min_snap == 0:
				gas_profile_all   = np.array([])
				dark_profile_all  = np.array([])
				type2_profile_all = np.array([])
				type3_profile_all = np.array([])
				type4_profile_all = np.array([])
			else:
				newfile = h5py.File(newfilename,'r')
				gas_profile_all   = np.array(newfile['gas'])
				dark_profile_all  = np.array(newfile['dark'])
				type2_profile_all = np.array(newfile['type2'])
				type3_profile_all = np.array(newfile['type3'])
				type4_profile_all = np.array(newfile['type4'])
				newfile.close()

			#---read-snapshots-and-calculate-------------------------------------------------------
			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
			dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

			x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
			dark_cm = np.array([x_cm,y_cm,z_cm]).T
			d_dark = np.linalg.norm(dark_pos-dark_cm, axis=1)

			gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
			gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
			d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)

			type2mass = snapHDF5.read_block(snapfile, 'MASS', parttype=2)*(1.e10)/h
			type2_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=2)/h
			d_type2 = np.linalg.norm(type2_pos-dark_cm, axis=1)

			type3mass = snapHDF5.read_block(snapfile, 'MASS', parttype=3)*(1.e10)/h
			type3_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=3)/h
			d_type3 = np.linalg.norm(type3_pos-dark_cm, axis=1)

			type4mass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
			type4_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h

			try:
				d_type4 = np.linalg.norm(type4_pos-dark_cm, axis=1)
				hastype4 = True
			except:
				hastype4 = False

			dark_profile,gas_profile,type2_profile,type3_profile,type4_profile = (np.array([]),)*5

			for dist in drange:
				dark_profile  = np.append(dark_profile, np.sum(darkmass[(d_dark < dist)]))
				gas_profile   = np.append(gas_profile,  np.sum(gasmass[(d_gas < dist)]))
				type2_profile = np.append(type2_profile,np.sum(type2mass[(d_type2 < dist)]))
				type3_profile = np.append(type3_profile,np.sum(type3mass[(d_type3 < dist)]))

				if hastype4: type4_profile = np.append(type4_profile,np.sum(type4mass[(d_type4 < dist)]))
				else: 		 type4_profile = np.append(type4_profile,0.)

			try:
				gas_profile_all   = np.vstack((gas_profile_all,gas_profile))
				dark_profile_all  = np.vstack((dark_profile_all,dark_profile))
				type2_profile_all = np.vstack((type2_profile_all,type2_profile))
				type3_profile_all = np.vstack((type3_profile_all,type3_profile))
				type4_profile_all = np.vstack((type4_profile_all,type4_profile))
			except:
				gas_profile_all   = gas_profile
				dark_profile_all  = dark_profile
				type2_profile_all = type2_profile
				type3_profile_all = type3_profile
				type4_profile_all = type4_profile

			#---overwrite-old-file-with-data-calculated-for-current-snapshot-----------------------
			newfile = h5py.File(newfilename,'w')
			newfile.create_dataset('drange',data=drange)
			newfile.create_dataset('gas',   data=gas_profile_all)
			newfile.create_dataset('dark',  data=dark_profile_all)
			newfile.create_dataset('type2', data=type2_profile_all)
			newfile.create_dataset('type3', data=type3_profile_all)
			newfile.create_dataset('type4', data=type4_profile_all)
			newfile.close()

		print('\nfinished')

	newfile.close()

	
	# print('---------------------------------------------------------------'+'\n')

def calculate_mass_profiles_temp(sim,hires=True):	
	'''
	this one ACTUALLY writes data after every snapshot so
	it isn't lost if the connection is dropped

	but it writes it to separate hdf5 keys and will need to be 
	consolidated before use
	'''
	if hires:
		newfilename = d.datdir+'massprofiles_'+sim+'_hires_temp.hdf5'
		drange = np.logspace(-2,2,2000)
	else:
		newfilename = d.datdir+'massprofiles_'+sim+'_temp.hdf5'
		drange = np.logspace(-1,np.log10(250),100)

	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))
	print('writing file after every snapshot to a separate hdf5 key in order to preserve data integrity')
	print('you will need to consolidate the data before it is usable')

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			keysview = newfile.keys()
			keys = np.array([])
			for k in keysview:
				keys = np.append(keys,k)
			keys.sort()
			min_snap = int(keys[-1][-3:])+1
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')
			newfile = h5py.File(newfilename,'w')

	except:
		min_snap = 0
		preexist = False
		print('file does not exist: creating')
		newfile = h5py.File(newfilename,'w')
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')
		dummy=''

	else:
		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		for snapnum in np.arange(min_snap,max_snap+1):
			#---print-current-snapshot-------------------------------------------------------------
			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))
			else:
				print('')

			#---read-snapshots-and-calculate-------------------------------------------------------
			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
			dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

			x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
			dark_cm = np.array([x_cm,y_cm,z_cm]).T
			d_dark = np.linalg.norm(dark_pos-dark_cm, axis=1)

			gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
			gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
			d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)

			type2mass = snapHDF5.read_block(snapfile, 'MASS', parttype=2)*(1.e10)/h
			type2_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=2)/h
			d_type2 = np.linalg.norm(type2_pos-dark_cm, axis=1)

			type3mass = snapHDF5.read_block(snapfile, 'MASS', parttype=3)*(1.e10)/h
			type3_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=3)/h
			d_type3 = np.linalg.norm(type3_pos-dark_cm, axis=1)

			type4mass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
			type4_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h

			try:
				d_type4 = np.linalg.norm(type4_pos-dark_cm, axis=1)
				hastype4 = True
			except:
				hastype4 = False

			dark_profile,gas_profile,type2_profile,type3_profile,type4_profile = (np.array([]),)*5

			for dist in drange:
				dark_profile  = np.append(dark_profile, np.sum(darkmass[(d_dark < dist)]))
				gas_profile   = np.append(gas_profile,  np.sum(gasmass[(d_gas < dist)]))
				type2_profile = np.append(type2_profile,np.sum(type2mass[(d_type2 < dist)]))
				type3_profile = np.append(type3_profile,np.sum(type3mass[(d_type3 < dist)]))

				if hastype4: type4_profile = np.append(type4_profile,np.sum(type4mass[(d_type4 < dist)]))
				else: 		 type4_profile = np.append(type4_profile,0.)

			#---write-current-snapshot-profiles-to-file--------------------
			newfile.create_dataset('gas_'+str(snapnum).zfill(3),   data=gas_profile)
			newfile.create_dataset('dark_'+str(snapnum).zfill(3),  data=dark_profile)
			newfile.create_dataset('type2_'+str(snapnum).zfill(3), data=type2_profile)
			newfile.create_dataset('type3_'+str(snapnum).zfill(3), data=type3_profile)
			newfile.create_dataset('type4_'+str(snapnum).zfill(3), data=type4_profile)
		
		newfile.close()
		print('\nfinished')

	newfile.close()	
	print('---------------------------------------------------------------'+'\n')

def calculate_mass_profiles_txt(sim):
	'''
	this one writes each mass profile to a separate text file

	holy shit why isn't any of this working
	'''

	drange = np.logspace(-2,2,2000)

	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	# print('writing to: '+str(newfilename))
	print('writing file after every snapshot to a separate txt file in order to preserve data integrity')
	print('you will need to consolidate the data before it is usable')

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	mypath = d.datdir+'massprofiles/'+sim
	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	if len(onlyfiles) > 0:
		a = np.sort(onlyfiles)
		min_snap = int(a[-1][:3]) + 1
	else:
		min_snap = 0

	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])
	# if sim=='rho0500_1e6':
	# 	max_snap = 275

	try:
		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()
	except:
		raise ValueError('could not find '+d.datdir+'centerofmass_'+sim+'.hdf5. please calculate centers of mass first.')

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		for snapnum in np.arange(min_snap,max_snap+1):
			#---print-current-snapshot-------------------------------------------------------------
			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			#---read-snapshots-and-calculate-------------------------------------------------------
			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
			dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

			x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
			cm_snap = np.array([x_cm,y_cm,z_cm]).T

			dark_cm = CoM_all[snapnum]

			d_dark = np.linalg.norm(dark_pos-dark_cm, axis=1)

			gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
			gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
			d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)

			type2mass = snapHDF5.read_block(snapfile, 'MASS', parttype=2)*(1.e10)/h
			type2_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=2)/h
			d_type2 = np.linalg.norm(type2_pos-dark_cm, axis=1)

			type3mass = snapHDF5.read_block(snapfile, 'MASS', parttype=3)*(1.e10)/h
			type3_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=3)/h
			d_type3 = np.linalg.norm(type3_pos-dark_cm, axis=1)

			type4mass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
			type4_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h

			try:
				d_type4 = np.linalg.norm(type4_pos-dark_cm, axis=1)
				hastype4 = True
			except:
				hastype4 = False

			dark_profile,gas_profile,type2_profile,type3_profile,type4_profile = (np.array([]),)*5

			for dist in drange:
				dark_profile  = np.append(dark_profile, np.sum(darkmass[(d_dark < dist)]))
				gas_profile   = np.append(gas_profile,  np.sum(gasmass[(d_gas < dist)]))
				type2_profile = np.append(type2_profile,np.sum(type2mass[(d_type2 < dist)]))
				type3_profile = np.append(type3_profile,np.sum(type3mass[(d_type3 < dist)]))

				if hastype4: type4_profile = np.append(type4_profile,np.sum(type4mass[(d_type4 < dist)]))
				else: 		 type4_profile = np.append(type4_profile,0.)

			#---write-data-to-text-file---column=different-data-------------
			data = np.vstack((drange,gas_profile,dark_profile,type2_profile,type3_profile,type4_profile))

			# if np.any(dark_profile==0):
			# 	n_z = np.count_nonzero(dark_profile==0)
			# 	id_z = np.where(dark_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in dark_profile! occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')

			# if np.any(gas_profile==0):
			# 	n_z = np.count_nonzero(gas_profile==0)
			# 	id_z = np.where(gas_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in gas_profile! occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')

			# if np.any(type2_profile==0):
			# 	n_z = np.count_nonzero(type2_profile==0)
			# 	id_z = np.where(type2_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in type2_profile! occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')

			# if np.any(type3_profile==0):
			# 	n_z = np.count_nonzero(type3_profile==0)
			# 	id_z = np.where(type3_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in type3_profile! occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')

			# if np.any(dark_profile==0) or np.any(gas_profile==0) or np.any(type2_profile==0) or np.any(type3_profile==0):
			# 	raise ValueError('fix those zeros!')

			# if np.any(type4_profile==0):
			# 	n_z = np.count_nonzero(type4_profile==0)
			# 	id_z = np.where(type4_profile==0)[0]
			# 	print('you got ' + str(n_z) + ' zeros in type4_profile, but that\'s actually possible, so we will proceed. occurred at snapshot ' + str(snapnum).zfill(3))
			# 	print('these zeros occurred at: ')
			# 	print(id_z)
			# 	print('the distances these corresponds to are:')
			# 	print(drange[id_z])
			# 	print('')
				
			fname = d.datdir+'massprofiles/'+sim + '/' + str(snapnum).zfill(3) +'.txt'
			np.savetxt(fname, data, delimiter=',')

		print('\nfinished')

	print('---------------------------------------------------------------'+'\n')

def consolidate_txt_files(sim):
	mypath = d.datdir+'massprofiles/'+sim

def calculate_sfgas_profile(sim):
	if not(sim in models):
		raise ValueError('please choose a sim in models')

	i = np.where(models==sim)[0][0]

	newfile = h5py.File(d.datdir+'massprofiles_'+savenames[i]+'.hdf5','r+')
	drange = np.logspace(-1,np.log10(250),100)

	for snapnum in range(401):
		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)

		print(snapfile)
		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
		dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

		x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
		dark_cm = np.array([x_cm,y_cm,z_cm]).T
		d_dark = np.linalg.norm(dark_pos-dark_cm, axis=1)

		gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
		gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
		gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
		d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)

		gasmass = gasmass[(gas_sfr > 0)]
		d_gas = d_gas[(gas_sfr > 0)]

		sfgas_profile = np.array([])

		for dist in drange:
			thismass = np.sum(gasmass[d_gas < dist])
			sfgas_profile = np.append(sfgas_profile,thismass)

		if snapnum==0:
			sfgas_profile_all = sfgas_profile
				
		else:
			sfgas_profile_all   = np.vstack((sfgas_profile_all,sfgas_profile))
		
	newfile.create_dataset('sfgas',data=sfgas_profile_all)
	newfile.close()

def calculate_Rhalfs(sim):
	newfilename = d.datdir+'rhalfs_'+sim+'.hdf5'
	
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))
	drange = np.logspace(-1,1,300)

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			all_Rhalf = np.array(newfile['Rhalf'])
			min_snap = all_Rhalf.shape[0]
			preexist = True
			print('file exists: opening')
		except:
			min_snap = 0
			preexist = False
			all_Rhalf = np.array([])
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		min_snap = 0
		preexist = False
		all_Rhalf = np.array([])
		print('file does not exist: creating')
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	allfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	snaplist = np.sort(allfiles[np.flatnonzero(np.core.defchararray.find(allfiles,'snapshot')!=-1)])
	max_snap = int(snaplist[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		for snapnum in range(min_snap,max_snap+1):
			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)

			# print(snapfile)
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))

			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
			dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

			x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
			dark_cm = np.array([x_cm,y_cm,z_cm]).T 

			type4mass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
			type4_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h - dark_cm

			totaltype4mass = np.sum(type4mass)

			try:
				# d_type4 = np.linalg.norm(type4_pos-dark_cm, axis=1)
				d_type4 = np.sqrt( (type4_pos[:,0]-dark_cm[0])**2 + (type4_pos[:,1]-dark_cm[1])**2   )
				hastype4 = True
			except:
				hastype4 = False

			type4_profile = np.array([])

			for dist in drange:
				if hastype4: 



					type4_profile = np.append(type4_profile,np.sum(type4mass[(d_type4 < dist)]))
				else: 		
					type4_profile = np.append(type4_profile,0.)

			this_Rhalf = drange[m.find_nearest(type4_profile,0.5*totaltype4mass,getindex=True)]
			all_Rhalf = np.append(all_Rhalf,this_Rhalf)
			
		#---write--------------------------------------------------------------------------------------
		if preexist:
			del newfile['Rhalf']

		# newfile.create_dataset('drange',data=drange)
		newfile.create_dataset('Rhalf',   data=all_Rhalf)
		print('\nfinished')

	newfile.close()
	print('---------------------------------------------------------------'+'\n')

def calculate_rho(sim):
	newfilename = d.datdir+'rho_list_all_'+sim+'.hdf5'
	print('---------------------------------------------------------------')
	print('calculating rho for: '+sim)
	print('writing to: '+str(newfilename))
	
	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			kl = np.sort(np.array(newfile.keys()))
			last_snap = int(kl[-1].split('_')[-1])
			min_snap = last_snap + 1
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'w')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	mypath = d.smuggledir+sim
	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = np.sort(onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)])
	max_snap = int(a[-1].split('.')[0][-3:])

	#----------------------------------------------------------------------------------------------

	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		rho_bins = np.logspace(-6,6,100)
		
		for snapnum in np.arange(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing); sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')
			
			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)

			rho = snapHDF5.read_block(snapfile,"RHO ",parttype=0)
			rho *= m.Xh/m.PROTONMASS*m.UnitDensity_in_cgs

			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
			dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
			x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
			dark_cm = np.array([x_cm,y_cm,z_cm]).T

			gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
			d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)
			rho = rho[d_gas < 1]

			dname = 'rho_1kpc_'+str(snapnum).zfill(3)
			newfile.create_dataset(dname,data=rho)

	newfile.close()

	
	print('---------------------------------------------------------------'+'\n')

def calculate_core_radius(sim,nfw_dcut=3,ratio_cut=2,hires=True):
	# if hires:	newfilename = d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'_ratio'+str(ratio_cut)+'_hires.hdf5'
	# else: 	    newfilename = d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'_ratio'+str(ratio_cut)+'.hdf5'
	if hires:	newfilename = d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'_hires.hdf5'
	else: 	    newfilename = d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'.hdf5'
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			core_radius = np.array(newfile['core_radius'])
			time = np.array(newfile['time'])
			min_snap = len(core_radius)
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False

	#----------------------------------------------------------------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap_sims = int(a[-1].split('.')[0][-3:])

	#---read-massprofiles-and-figure-out-ending-place----------------------------------------
	if hires: f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
	else:     f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	dark_profile_all = np.array(f['dark'])
	f.close()

	max_snap = dark_profile_all.shape[0]-1

	if max_snap < max_snap_sims:
		print('massprofiles_'+sim+'.hdf5 needs to be updated. currently calculated mass profiles up to '+str(max_snap)+', but simulation has progressed to '+str(max_snap_sims))
		# calculate_mass_profiles(sim,hires=hires)

	if min_snap==0:
		core_radius = np.array([])
		time = np.array([])

	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		print('calculating with nfw_dcut='+str(np.round(nfw_dcut,2))+' and ratio_cut='+str(np.round(ratio_cut,2)))

		for snapnum in np.arange(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing); sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			
			header = snapHDF5.snapshot_header(snapfile)
			time = np.append(time, header.time/0.7)

			vols = 4./3.*np.pi*(drange**3)
			density = dark_profile_all[snapnum]/vols

			#---fit-NFW-to-current-snapshot----------------------------------------------------
			H = 70./1000.
			rho_crit = 3*H**2 / (8*np.pi*m.Gprime) 
			r200 = drange[(density >= 200*rho_crit)][-1]

			rho_DM_fit = density[(drange > nfw_dcut)]
			drange_fit = drange[(drange > nfw_dcut)]
			fitted_pars = fit.NFW_sqfit(drange_fit,rho_DM_fit,c_guess=10,rho0_guess=1e7,r200=r200)
			density_nfw = fit.NFW_model(fitted_pars,drange,r200)

			#---select-reference-profile-------------------------------------------------------
			# density_ref = dark_profile_all[0]/vols
			density_ref = density_nfw
		
			#---find-where-current-density-deviates-from-reference-profile-by-a-factor-of-two--
			seld = (drange < 10) & (density > 0)
			rho_ratio = density_ref[seld] / density[seld]
			sel_in_2 = (rho_ratio > ratio_cut)# & (rho_ratio < 1.8)

			if np.count_nonzero(sel_in_2)==0:
				core_radius = np.append(core_radius,0.)
			else:
				core_radius = np.append(core_radius,drange[seld][np.where(sel_in_2)[0][-1]])
		
		print('writing to file')
		if preexist:
			del newfile['time'];del newfile['core_radius'];del newfile['nfw_dcut'];del newfile['ratio_cut']
		
		newfile.create_dataset('time',data=time)
		newfile.create_dataset('core_radius',data=core_radius)
		newfile.create_dataset('nfw_dcut',data=nfw_dcut)
		newfile.create_dataset('ratio_cut',data=ratio_cut)
		newfile.close()

	print('---------------------------------------------------------------'+'\n')

def calculate_powerlaw_slope(sim,nfw_dcut=3,hires=True,fixed_dist=False):
	'''
	automatically calculates with Power convergence radius
	and core radius as min and max bounds on denisty profile
	(respectively) if using the hires flag

	will calculate between d=0.1 and d=0.3kpc otherwise
	'''
	if hires:
		if fixed_dist:
			newfilename = d.datdir+'powerlawslope_nfwdcut'+str(nfw_dcut)+'_'+sim+'_hires_fixed.hdf5'
		else:
			newfilename = d.datdir+'powerlawslope_nfwdcut'+str(nfw_dcut)+'_'+sim+'_hires.hdf5'
	else:
		newfilename = d.datdir+'powerlawslope_nfwdcut'+str(nfw_dcut)+'_'+sim+'.hdf5'
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			slope = np.array(newfile['slope'])
			time = np.array(newfile['time'])
			min_snap = len(slope)
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False

	#----------------------------------------------------------------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap_sims = int(a[-1].split('.')[0][-3:])

	#---read-massprofiles-and-figure-out-ending-place----------------------------------------
	if hires: f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')			
	else:     f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	dark_profile_all = np.array(f['dark'])
	f.close()

	vols = 4./3.*np.pi*(drange**3)
	max_snap = dark_profile_all.shape[0]-1

	if max_snap < max_snap_sims:
		print('massprofiles for '+sim+' needs to be updated. currently calculated mass profiles up to '+str(max_snap)+', but simulation has progressed to '+str(max_snap_sims))
		# calculate_mass_profiles(sim,hires=hires)

	if min_snap==0:
		slope = np.array([])
		time = np.array([])

	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		if hires:
			nfname = d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'_hires.hdf5'
			nf = h5py.File(nfname,'r')
			core_radius = np.array(nf['core_radius'])
			nf.close()

			# pfname = d.datdir+'power_radius_'+sim+'.hdf5'
			# pf = h5py.File(pfname,'r')
			# power_radius = np.array(pf['power_radius'])
			# pf.close()

		for snapnum in np.arange(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing); sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			header = snapHDF5.snapshot_header(snapfile)
			time = np.append(time, header.time/0.7)
			rho_dm = dark_profile_all[snapnum] / vols

			# if hires: sel_fit = (drange > power_radius[snapnum]) & (drange < core_radius[snapnum])
			# else:     sel_fit = (drange > 0.1) & (drange < 0.3)

			if hires:
				if fixed_dist:
					sel_fit = (drange > 0.05) & (drange < 0.55)			
				else:
					# sel_fit = (drange > power_radius[snapnum]) & (drange < core_radius[snapnum])
					sel_fit = (drange > 0.05) & (drange < core_radius[snapnum])
			else:
				sel_fit = (drange > 0.1) & (drange < 0.3)
			# sel_fit = (drange > 0.1) & (drange < 0.3)

			
			rho_dm_fit = rho_dm[sel_fit]
			drange_fit = drange[sel_fit]	
			fitted_pars = fit.powerlaw_sqfit(xdata=drange_fit,ydata=rho_dm_fit,a_guess=1e8,k_guess=-0.1)
			k = fitted_pars[1]
			slope = np.append(slope,k)			
		
		print('writing to file')
		if preexist:
				del newfile['time'];del newfile['slope']
		
		newfile.create_dataset('time',data=time)
		newfile.create_dataset('slope',data=slope)
		newfile.close()

	print('---------------------------------------------------------------'+'\n')

def calculate_powerlaw_slope_PITS(sim,hires=True,fixed_dist=False):
	'''
	automatically calculates with Power convergence radius
	and core radius as min and max bounds on denisty profile
	(respectively) if using the hires flag

	will calculate between d=0.1 and d=0.3kpc otherwise
	'''
	if fixed_dist and hires:
		newfilename = d.datdir+'PITSslope_'+sim+'_hires_fixed.hdf5'
	if not(fixed_dist) and hires:
		newfilename = d.datdir+'PITSslope_'+sim+'_hires.hdf5'
	if not(hires):
		newfilename = d.datdir+'PITSslope_'+sim+'.hdf5'

	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			slope = np.array(newfile['slope'])
			time = np.array(newfile['time'])
			min_snap = len(slope)
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False

	#----------------------------------------------------------------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap_sims = int(a[-1].split('.')[0][-3:])

	#---read-massprofiles-and-figure-out-ending-place----------------------------------------
	if hires: f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')			
	else:     f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	dark_profile_all = np.array(f['dark'])
	f.close()

	if hires:	pf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5')
	else:		pf = h5py.File(d.datdir+'PITS_params_'+sim+'.hdf5')
	rcore_PITS = np.array(pf['rc'])
	pf.close()

	vols = 4./3.*np.pi*(drange**3)
	max_snap = dark_profile_all.shape[0]-1

	if max_snap < max_snap_sims:
		print('massprofiles for '+sim+' needs to be updated. currently calculated mass profiles up to '+str(max_snap)+', but simulation has progressed to '+str(max_snap_sims))
		# calculate_mass_profiles(sim,hires=hires)

	if min_snap==0:
		slope = np.array([])
		time = np.array([])

	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))


		if hires and not(fixed_dist):
			nf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5','r')
			core_radius = np.array(nf['rc'])
			nf.close()

		if hires:
			pf = h5py.File(d.datdir+'power_radius_'+sim+'.hdf5','r')
			power_radius = np.array(pf['power_radius'])
			pf.close()
			
		for snapnum in np.arange(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing); sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			header = snapHDF5.snapshot_header(snapfile)
			time = np.append(time, header.time/0.7)
			rho_dm = dark_profile_all[snapnum] / vols

			if hires:
				if fixed_dist:
					sel_fit = (drange > 0.05) & (drange < 0.556942176925)
				else:
					sel_fit = (drange > power_radius[snapnum]) & (drange < core_radius[snapnum])
			else:
				sel_fit = (drange > 0.1) & (drange < 0.556942176925)

			rho_dm_fit = rho_dm[sel_fit]
			drange_fit = drange[sel_fit]	
			fitted_pars = fit.powerlaw_sqfit(xdata=drange_fit,ydata=rho_dm_fit,a_guess=1e8,k_guess=-0.1)
			k = fitted_pars[1]
			slope = np.append(slope,k)			
		
		print('writing to file')
		if preexist:
				del newfile['time'];del newfile['slope']
		
		newfile.create_dataset('time',data=time)
		newfile.create_dataset('slope',data=slope)
		newfile.close()

	print('---------------------------------------------------------------'+'\n')

def calculate_power_radius(sim):
	newfilename = d.datdir+'power_radius_'+sim+'.hdf5'
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			power_radius = np.array(newfile['power_radius'])
			time = np.array(newfile['time'])
			min_snap = len(power_radius)
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False

	#----------------------------------------------------------------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---figure-out-ending-place--------------------------------------------------------------------
	if min_snap==0:
		power_radius = np.array([])
		time = np.array([])

	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		for snapnum in np.arange(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing); sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			header = snapHDF5.snapshot_header(snapfile)
			time = np.append(time, header.time/0.7)

			darkpos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
			darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h

			x_cm = np.sum(darkpos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(darkpos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(darkpos[:,2] * darkmass) / np.sum(darkmass)
			cm = np.array([x_cm,y_cm,z_cm]).T

			darkpos = darkpos - cm

			d_dark = np.linalg.norm(darkpos,axis=1)
			dist_ordered = np.sort(d_dark)

			d_200 = dist_ordered[199]

			power_radius = np.append(power_radius,d_200)
		
		print('writing to file')
		if preexist:
				del newfile['time'];del newfile['power_radius']
		
		newfile.create_dataset('time',data=time)
		newfile.create_dataset('power_radius',data=power_radius)
		newfile.close()

	print('---------------------------------------------------------------'+'\n')

def convert_sfr(sim):
	filein = d.smuggledir + sim + '/sfr.txt'
	print(filein)
	data = np.loadtxt(filein)
	time = data[:,0]/h
	sfr = data[:,2]
	sfr[(sfr == 0)]=1.e-10

	newfile = h5py.File('sfr_time_'+sim+'.hdf5','w')
	newfile.create_dataset('time',data=time)
	newfile.create_dataset('sfr',data=sfr)
	newfile.close()
	
def measure_single_rcore(sim,snapnum,nfw_dcut=3,ratio_cut=2,hires=False):
	if hires: f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
	else:     f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	dark_profile_all = np.array(f['dark'])
	f.close()

	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
	
	vols = 4./3.*np.pi*(drange**3)
	density = dark_profile_all[snapnum]/vols

	#---fit-NFW-to-current-snapshot----------------------------------------------------
	H = 70./1000.
	rho_crit = 3*H**2 / (8*np.pi*m.Gprime) 
	r200 = drange[(density >= 200*rho_crit)][-1]
	rho_DM_fit = density[(drange > nfw_dcut)]
	drange_fit = drange[(drange > nfw_dcut)]
	fitted_pars = fit.NFW_sqfit(drange_fit,rho_DM_fit,c_guess=10,rho0_guess=1e7,r200=r200)
	density_nfw = fit.NFW_model(fitted_pars,drange,r200)

	#---select-reference-profile-------------------------------------------------------
	# density_ref = dark_profile_all[0]/vols
	density_ref = density_nfw

	#---find-where-current-density-deviates-from-reference-profile-by-a-factor-of-two--
	seld = (drange < 10) & (density > 0)
	rho_ratio = density_ref[seld] / density[seld]
	sel_in_2 = (rho_ratio > ratio_cut)# & (rho_ratio < 1.8)

	if np.count_nonzero(sel_in_2)==0:
		core_radius = 0
	else:
		core_radius = drange[seld][np.where(sel_in_2)[0][-1]]

	print('rcore for '+sim+' at snapshot '+str(snapnum)+' = '+str(np.round(core_radius,5))+' kpc')

def measure_single_slope(sim,snapnum,rfit_max=0.3,hires=1):
	if hires:
		f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
	else:
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	vols = 4./3.*np.pi*(drange**3)
	dark_profile_all = np.array(f['dark'])
	f.close()

	sel_fit = (drange > 0.1) & (drange < rfit_max)
	drange_fit = drange[sel_fit]

	rho_dm = dark_profile_all[snapnum] / vols
	rho_dm_fit = rho_dm[sel_fit]
	fitted_pars = fit.powerlaw_sqfit(xdata=drange_fit,ydata=rho_dm_fit,a_guess=1e8,k_guess=-0.1)
	slope = fitted_pars[1]

	print('slope for '+sim+' at snapshot '+str(snapnum).zfill(3)+' = '+str(np.round(slope,5)))
	
# def calculate_rho_frac_profile(sim,snapnum):

def trace_particle_positions(sim):
	fname = d.datdir+'part_positions_'+sim+'.txt'
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(fname))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		all_data = np.loadtxt(fname,dtype=float,delimiter=',')
		trace_ids = all_data[0]
		dist_arr = all_data[1:]
		min_snap = dist_arr.shape[0]
		print('successfully read from file')
	except:
		min_snap = 0
		print('could not read from file, starting from scratch')

	mypath = d.smuggledir+sim
	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	if min_snap>=max_snap:
		print('read all available data. exiting.')
	else:
		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		try:
			cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
			CoM_all = np.array(cm_file['CoM'])
			cm_file.close()
		except:
			raise ValueError('could not find '+d.datdir+'centerofmass_'+sim+'.hdf5. please calculate center of masses before tracing particle positions')
	
		# read initial snapshot and pick particles that start at a variety of initial radii
		if min_snap==0:
			print('initialization: determining which particles to trace')
			initfile = d.smuggledir+sim+'/snapshot_'+str(0).zfill(3)	
			init_distances = np.array([0.1,0.2,0.5,1,2,5])
			cm_0 = CoM_all[0]
						
			dark_id_all = snapHDF5.read_block(initfile, 'ID  ', parttype=1)
			darkpos_0 = snapHDF5.read_block(initfile, 'POS ', parttype=1)/h
			dist_0_all = np.linalg.norm(darkpos_0 - cm_0,axis=1)

			trace_ids = np.array([])

			for dist in init_distances:
				idist = m.find_nearest(dist_0_all,dist,getindex=1)
				trace_ids = np.append(trace_ids,dark_id_all[idist])	

		print('tracing: ',trace_ids)

		# loop through snapshots and find distance of each particle
		for snapnum in np.arange(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap-1): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)	
			dark_id_all = snapHDF5.read_block(snapfile, 'ID  ', parttype=1)
			darkpos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
			cm = CoM_all[snapnum]	

			this_snap_dists = np.array([])

			for trace_id in trace_ids:
				location = np.where(dark_id_all==trace_id)[0][0]
				this_snap_dists = np.append(this_snap_dists,np.linalg.norm(darkpos[location] - cm))

			if snapnum==0:
				dist_arr = this_snap_dists
			else:
				dist_arr = np.vstack((dist_arr,this_snap_dists))

	# add particle IDs to top of dist_arr if they aren't already there
	if min_snap==0:	
		dist_arr = np.vstack((trace_ids,dist_arr))

	np.savetxt(fname, dist_arr, delimiter=',')
	print('\nfinished')
	print('---------------------------------------------------------------')

def calculate_r50SFgas(sim,do_weight=False):
	if do_weight:
		newfilename = d.datdir+'r50SFgas_massweight_'+sim+'.hdf5'
	else:
		newfilename = d.datdir+'r50SFgas_'+sim+'.hdf5'
	
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))
	drange = np.logspace(-1,1,300)

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			all_r50 = np.array(newfile['all_r50'])
			min_snap = all_Rhalf.shape[0]
			preexist = True
			print('file exists: opening')
		except:
			min_snap = 0
			preexist = False
			all_r50 = np.array([])
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		min_snap = 0
		preexist = False
		all_r50 = np.array([])
		print('file does not exist: creating')
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	allfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	snaplist = np.sort(allfiles[np.flatnonzero(np.core.defchararray.find(allfiles,'snapshot')!=-1)])
	max_snap = int(snaplist[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		try:
			cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
			CoM_all = np.array(cm_file['CoM'])
			cm_file.close()
		except:
			raise ValueError('please calculate center of mass before proceeding')

		for snapnum in range(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))

			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h - CoM_all[snapnum]
			gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
			gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
			
			sel_sf = (gas_sfr > 0)

			gas_pos = gas_pos[sel_sf]
			gas_sfr = gas_sfr[sel_sf]

			r_gas = np.linalg.norm(gas_pos,axis=1) 

			dmax = 10
			drange = np.logspace(-1,np.log10(dmax),2000)
			sfr_cumulative = np.array([])

			for dist in drange:
				sfr_cumulative = np.append(sfr_cumulative,np.sum(gas_sfr[r_gas < dist]))

			ihalf = m.find_nearest(sfr_cumulative,0.5*np.amax(sfr_cumulative),getindex=1)
			rhalf = drange[ihalf]

			if do_weight:
				sfgas_mass = np.sum(gasmass[sel_sf])
				allgas_mass = np.sum(gasmass)
				frac = sfgas_mass / allgas_mass
				rhalf = rhalf * frac
			
			all_r50 = np.append(all_r50,rhalf)


		#---write--------------------------------------------------------------------------------------
		if preexist:
			del newfile['all_r50']

		newfile.create_dataset('all_r50', data=all_r50)
		print('\nfinished')

	newfile.close()
	print('---------------------------------------------------------------'+'\n')

def calculate_r50young(sim,age_cut=50):
	newfilename = d.datdir+'r50young_'+str(age_cut)+'_'+sim+'.hdf5'
	
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))
	drange = np.logspace(-1,1,300)

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			all_r50 = np.array(newfile['all_r50'])
			min_snap = all_Rhalf.shape[0]
			preexist = True
			print('file exists: opening')
		except:
			min_snap = 0
			preexist = False
			all_r50 = np.array([])
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		min_snap = 0
		preexist = False
		all_r50 = np.array([])
		print('file does not exist: creating')
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	allfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	snaplist = np.sort(allfiles[np.flatnonzero(np.core.defchararray.find(allfiles,'snapshot')!=-1)])
	max_snap = int(snaplist[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		try:
			cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
			CoM_all = np.array(cm_file['CoM'])
			cm_file.close()
			tf = h5py.File(d.datdir+'timefile.hdf5','r')
			time = np.array(tf['time'])
			tf.close()
		except:
			raise ValueError('please calculate center of mass before proceeding')

		for snapnum in range(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))

			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			star_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h - CoM_all[snapnum]
			starmass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
			# star_age = snapHDF5.read_block(snapfile, 'AGE ', parttype=4)/h * 1.e3 # now in Myr
			star_age = (time[snapnum] - (snapHDF5.read_block(snapfile, 'AGE ', parttype=4)/h)) * 1.e3 # now in Myr
			
			if not(type(starmass)==np.ndarray):
				starmass = np.array([starmass])

			if len(starmass) > 5: 
				sel = (star_age < age_cut)

				if np.count_nonzero(sel) > 5:
					star_pos = star_pos[sel]
					starmass = starmass[sel]

					d_star = np.linalg.norm(star_pos,axis=1) 
					m_total = np.sum(starmass)

					# for dist in np.logspace(-2,1,2000):
					# 	mass_in_dist = np.sum(starmass[(d_star < dist)])
					# 	err_percent = (mass_in_dist - 0.5*m_total)/(0.5*m_total) * 100.0
					# 	if err_percent < 1:
					# 		this_Rhalf = dist
					# 		break
					# all_r50 = np.append(all_r50,this_Rhalf)

					profile = np.array([])
					drange = np.logspace(-2,1,2000)

					for dist in drange:
						profile = np.append(profile,np.sum(starmass[d_star < dist]))

					ihalf = m.find_nearest(profile,0.5*m_total,getindex=1)
					rhalf = drange[ihalf]
					
					all_r50 = np.append(all_r50,rhalf)

					# print('at snapshot '+str(snapnum).zfill(3)+', r50 = '+str(np.round(rhalf,5)))
				else:
					# print('no stars < '+str(age_cut)+' Myr old in snapshot '+str(snapnum))
					all_r50 = np.append(all_r50,0.)

			else:
				# print('not enough stars in snapshot '+str(snapnum))
				all_r50 = np.append(all_r50,0.)


			

		#---write--------------------------------------------------------------------------------------
		if preexist:
			del newfile['all_r50']

		newfile.create_dataset('all_r50', data=all_r50)
		print('\nfinished')

	newfile.close()
	print('---------------------------------------------------------------'+'\n')

def calculate_rmedianyoung(sim,age_cut=50):
	newfilename = d.datdir+'rmedianyoung_'+str(age_cut)+'_'+sim+'.hdf5'
	
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))
	drange = np.logspace(-1,1,300)

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			all_r50 = np.array(newfile['all_rmedian'])
			min_snap = all_Rhalf.shape[0]
			preexist = True
			print('file exists: opening')
		except:
			min_snap = 0
			preexist = False
			all_r50 = np.array([])
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		min_snap = 0
		preexist = False
		all_r50 = np.array([])
		print('file does not exist: creating')
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	allfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	snaplist = np.sort(allfiles[np.flatnonzero(np.core.defchararray.find(allfiles,'snapshot')!=-1)])
	max_snap = int(snaplist[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		try:
			cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
			CoM_all = np.array(cm_file['CoM'])
			cm_file.close()
			tf = h5py.File(d.datdir+'timefile.hdf5','r')
			time = np.array(tf['time'])
			tf.close()
		except:
			raise ValueError('please calculate center of mass before proceeding')

		for snapnum in range(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))

			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			star_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h - CoM_all[snapnum]
			starmass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
			star_age = (time[snapnum] - (snapHDF5.read_block(snapfile, 'AGE ', parttype=4)/h)) * 1.e3 # now in Myr
			
			if not(type(starmass)==np.ndarray):
				starmass = np.array([starmass])

			if len(starmass) > 5: 
				sel = (star_age < age_cut)

				if np.count_nonzero(sel) > 5:

					d_star = np.linalg.norm(star_pos[sel],axis=1) 
					r_median = np.median(d_star)
					all_r50 = np.append(all_r50, r_median)
				else:
					# print('no stars < '+str(age_cut)+' Myr old in snapshot '+str(snapnum))
					all_r50 = np.append(all_r50,0.)

			else:
				# print('not enough stars in snapshot '+str(snapnum))
				all_r50 = np.append(all_r50,0.)


			

		#---write--------------------------------------------------------------------------------------
		if preexist:
			del newfile['all_rmedian']

		newfile.create_dataset('all_rmedian', data=all_r50)
		print('\nfinished')

	newfile.close()
	print('---------------------------------------------------------------'+'\n')

def calculate_SFR_profile(sim):
	newfilename = d.datdir+'SFRprofile_'+sim+'.hdf5'
	drange = np.logspace(-2,2,2000)

	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			SFR_profile_all = np.array(newfile['SFR_profile_all'])
			min_snap = gas_profile_all.shape[0]
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')
		dummy=''

	else:
		try:
			cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
			CoM_all = np.array(cm_file['CoM'])
			cm_file.close()
		except:
			raise ValueError('please calculate center of mass before proceeding')

		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		for snapnum in range(min_snap,max_snap+1):
			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h - CoM_all[snapnum]
			gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)

			gas_pos = gas_pos[(gas_sfr > 0)]
			gas_sfr = gas_sfr[(gas_sfr > 0)]

			r_gas = np.linalg.norm(gas_pos,axis=1) 

			sfr_profile = np.array([])

			for dist in drange:
				sfr_profile = np.append(sfr_profile,np.sum(gas_sfr[r_gas < dist]))

			if snapnum==0:
				SFR_profile_all = sfr_profile
			else:
				SFR_profile_all = np.vstack((SFR_profile_all,sfr_profile))

		#---write--------------------------------------------------------------------------------------
		if preexist:
			del newfile['SFR_profile_all']; del newfile['drange']

		newfile.create_dataset('drange',data=drange)
		newfile.create_dataset('SFR_profile_all',data=SFR_profile_all)
		# print('\nfinished')

	newfile.close()

	
	# print('---------------------------------------------------------------'+'\n')

def consolidate_split_hdf5(sim):
	fname = d.datdir + 'massprofiles_' + sim + '_hires_temp.hdf5'
	f = h5py.File(fname,'r')
	k = np.array(f.keys())
	kl = np.sort(k)
	min_snap_temp = int(kl[0].split('_')[-1]) 
	max_snap_temp = int(kl[-1].split('_')[-1]) 
	print('found split hdf5 file, with data in separate keys from snapshot '+str(min_snap_temp)+' up to '+str(max_snap_temp))

	for snapnum in np.arange(min_snap_temp,max_snap_temp+1):
		if snapnum==min_snap_temp:
			gas_profile_all = np.array(f['gas_'+str(snapnum).zfill(3)])
			dark_profile_all = np.array(f['dark_'+str(snapnum).zfill(3)])
			type2_profile_all = np.array(f['type2_'+str(snapnum).zfill(3)])
			type3_profile_all = np.array(f['type3_'+str(snapnum).zfill(3)])
			type4_profile_all = np.array(f['type4_'+str(snapnum).zfill(3)])

		else:
			gas_profile_all = np.vstack((gas_profile_all,np.array(f['gas_'+str(snapnum).zfill(3)])))
			dark_profile_all = np.vstack((dark_profile_all,np.array(f['dark_'+str(snapnum).zfill(3)])))
			type2_profile_all = np.vstack((type2_profile_all,np.array(f['type2_'+str(snapnum).zfill(3)])))
			type3_profile_all = np.vstack((type3_profile_all,np.array(f['type3_'+str(snapnum).zfill(3)])))
			type4_profile_all = np.vstack((type4_profile_all,np.array(f['type4_'+str(snapnum).zfill(3)])))

	fname = d.datdir + 'massprofiles_' + sim + '_hires_fromtemp.hdf5'
	f = h5py.File(fname,'w')
	print('writing data to new standard file')
	f.create_dataset('drange',data = np.logspace(-2,2,2000))
	f.create_dataset('gas',data = gas_profile_all)
	f.create_dataset('dark',data = dark_profile_all)
	f.create_dataset('type2',data = type2_profile_all)
	f.create_dataset('type3',data = type3_profile_all)
	f.create_dataset('type4',data = type4_profile_all)
	f.close()

def consolidate_txt(sim):
	# fname = d.datdir + 'massprofiles_' + sim + '_hires_fromtemp.hdf5'
	mypath = d.datdir+'massprofiles/'+sim
	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = np.sort(onlyfiles)
	min_snap_txt = int(a[0][:3])
	max_snap_txt = int(a[-1][:3])
	print('found text files, looping through to consolidate data from snapshot '+str(min_snap_txt)+' up to '+str(max_snap_txt))

	for snapnum in np.arange(min_snap_txt,max_snap_txt+1):
		fname = mypath +'/' + str(snapnum).zfill(3)+'.txt'
		data = np.loadtxt(fname,delimiter=',',dtype=float)
		drange        = data[0]
		gas_profile   = data[1]
		dark_profile  = data[2]
		type2_profile = data[3]
		type3_profile = data[4]
		type4_profile = data[5]

		if snapnum==min_snap_txt:
			gas_profile_all   = gas_profile  
			dark_profile_all  = dark_profile 
			type2_profile_all = type2_profile
			type3_profile_all = type3_profile
			type4_profile_all = type4_profile
		else:
			gas_profile_all   = np.vstack((gas_profile_all  , gas_profile))
			dark_profile_all  = np.vstack((dark_profile_all , dark_profile))
			type2_profile_all = np.vstack((type2_profile_all, type2_profile))
			type3_profile_all = np.vstack((type3_profile_all, type3_profile))
			type4_profile_all = np.vstack((type4_profile_all, type4_profile))

	# now that the txt files have been read, try to find the hdf5 file that they should combine with
	# try:
	# 	fname = d.datdir + 'massprofiles_' + sim + '_hires_fromtemp.hdf5'
	# 	f = h5py.File(fname,'r')
	# 	preexist=True
	# 	print('found '+fname)
	# except:
	# 	try:
	# 		fname = d.datdir + 'massprofiles_' + sim + '_hires.hdf5'
	# 		f = h5py.File(fname,'r')
	# 		preexist=True
	# 		print('found '+fname)
	# 	except:
	# 		preexist=False
	# 		print('could not find previous hdf5 file')
	# if preexist:
	# 	gas_profile_old = np.array(f['gas'])
	# 	dark_profile_old = np.array(f['dark'])
	# 	type2_profile_old = np.array(f['type2'])
	# 	type3_profile_old = np.array(f['type3'])
	# 	type4_profile_old = np.array(f['type4'])
	# 	gas_profile_all = np.vstack((gas_profile_old, gas_profile_all))
	# 	dark_profile_all = np.vstack((dark_profile_old, dark_profile_all))
	# 	type2_profile_all = np.vstack((type2_profile_old, type2_profile_all))
	# 	type3_profile_all = np.vstack((type3_profile_old, type3_profile_all))
	# 	type4_profile_all = np.vstack((type4_profile_old, type4_profile_all))
	# 	newfilename = d.datdir + 'massprofiles_' + sim + '_hires_combined.hdf5'
	# else:

	newfilename = d.datdir + 'massprofiles_' + sim + '_hires.hdf5'

	# now write them to new file

	f = h5py.File(newfilename,'w')
	print('writing data to '+newfilename)
	f.create_dataset('drange',data = np.logspace(-2,2,2000))
	f.create_dataset('gas',data = gas_profile_all)
	f.create_dataset('dark',data = dark_profile_all)
	f.create_dataset('type2',data = type2_profile_all)
	f.create_dataset('type3',data = type3_profile_all)
	f.create_dataset('type4',data = type4_profile_all)
	f.close()

def calculate_PITS_params(sim,hires=1): #dcut
	if hires:
		newfilename = d.datdir+'PITS_params_'+sim+'_hires.hdf5'
	else:
		newfilename = d.datdir+'PITS_params_'+sim+'.hdf5'
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			rc_all = np.array(newfile['rc'])
			rho0_all = np.array(newfile['rho0'])
			min_snap = len(rc_all)
			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False

	#----------------------------------------------------------------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap_sims = int(a[-1].split('.')[0][-3:])

	#---read-massprofiles-and-figure-out-ending-place----------------------------------------
	if hires: f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')			
	else:     f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	dark_profile_all = np.array(f['dark'])
	f.close()

	sel_fit = (drange < 200)
	drange_fit = drange[sel_fit]

	vols = 4./3.*np.pi*(drange**3)
	max_snap = dark_profile_all.shape[0]-1

	if max_snap < max_snap_sims:
		print('massprofiles for '+sim+' needs to be updated. currently calculated mass profiles up to '+str(max_snap)+', but simulation has progressed to '+str(max_snap_sims))
		# calculate_mass_profiles(sim,hires=hires)

	if min_snap==0:
		rc_all = np.array([])
		rho0_all = np.array([])

	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		for snapnum in np.arange(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing); sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			rho_dm = dark_profile_all[snapnum] / vols
			rho_dm_fit = rho_dm[sel_fit]
			fitted_pars = fit.pits_sqfit(xdata=drange_fit,ydata=rho_dm_fit,rho0_guess=1e7,rc_guess=1)

			rho0 = fitted_pars[0]
			rc = fitted_pars[1]

			rho0_all = np.append(rho0_all,rho0)
			rc_all = np.append(rc_all,rc)
		
		print('writing to file')
		if preexist:
			del newfile['rho0'];del newfile['rc']
		
		newfile.create_dataset('rho0',data=rho0_all)
		newfile.create_dataset('rc',data=rc_all)
		newfile.close()

	print('---------------------------------------------------------------'+'\n')


def calculate_beta_anisotropy(sim,ptype='DM'):
	# newfilename = d.datdir+'beta_anisotropy_'+ptype+'_'+sim+'.hdf5'

	write_dir = d.datdir + 'beta_anisotropy_'+ptype+'_'+sim+'/'
	
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing txt files to: '+write_dir)
	# drange = np.logspace(-1,1,300)

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	# try:
	# 	newfile = h5py.File(newfilename,'r+')

	# 	try:
	# 		beta = np.array(newfile['beta'])
	# 		min_snap = beta.shape[0]
	# 		preexist = True
	# 		print('file exists: opening')
	# 	except:
	# 		min_snap = 0
	# 		preexist = False
	# 		beta = np.array([])
	# 		print('file exists, but is empty. starting from scratch.')

	# except:
	# 	newfile = h5py.File(newfilename,'x')
	# 	min_snap = 0
	# 	preexist = False
	# 	beta = np.array([])
	# 	print('file does not exist: creating')


		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	allfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	snaplist = np.sort(allfiles[np.flatnonzero(np.core.defchararray.find(allfiles,'snapshot')!=-1)])
	max_snap = int(snaplist[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap)+'\nnow reading up to: '+str(max_snap))

		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()

		v_cm_file = h5py.File(d.datdir+'centerofmassvelocity_'+sim+'.hdf5','r')
		v_cm_all = np.array(v_cm_file['Vel_CoM'])
		v_cm_file.close()

		for snapnum in range(min_snap,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))

			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			pos_all = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h - CoM_all[snapnum]
			vel_all = snapHDF5.read_block(snapfile, 'VEL ', parttype=1) - v_cm_all[snapnum]
			
			dist = np.linalg.norm(pos_all,axis=1)
			r_hat = pos_all / dist

			v_mag = np.linalg.norm(vel_all,axis=1)
			vr = np.diag(np.dot(r_hat,vel_all.T))
			
			v_tan = np.sqrt(np.abs(vr**2 -  v_mag**2))

			beta_this = 1 - ( v_tan**2 / (2*vr**2) )

			if snapnum == 0:
				beta = beta_this
			else:
				beta = np.vstack((beta,beta_this))

		#---write--------------------------------------------------------------------------------------
		if preexist:
			del newfile['beta']

		newfile.create_dataset('beta', data=beta)
		print('\nfinished')

	newfile.close()
	print('---------------------------------------------------------------'+'\n')

def calculate_vr_profile(ptype,sim):
	newfilename = d.datdir+'vr_allparticles_'+ptype+'_'+sim+'.hdf5'
	dist_bins = np.logspace(-1,1,100)

	if not(ptype in ['gas','dark','star']):
		raise ValueError('please choose ptype = gas, dark, or star')
	
	if ptype=='gas': parttype = 0
	elif ptype=='dark': parttype = 1
	elif ptype=='star': parttype = 4

	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			# vr_profile_all = np.array(newfile['vr_profile_all'])
			# min_snap = vr_profile_all.shape[0]
			k = np.array(newfile.keys())
			kl = np.sort(k)
			min_snap = int(kl[-1]) + 1
			print('file exists: opening')

			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')

	except:
		newfile = h5py.File(newfilename,'x')
		print('file does not exist: creating')
		min_snap = 0
		preexist = False
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	from os import listdir
	from os.path import isfile, join

	mypath = d.smuggledir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')
		dummy=''

	else:
		try:
			cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
			CoM_all = np.array(cm_file['CoM'])
			cm_file.close()

			v_cm_file = h5py.File(d.datdir+'centerofmassvelocity_'+sim+'.hdf5','r')
			v_cm_all = np.array(v_cm_file['Vel_CoM'])
			v_cm_file.close()
		except:
			raise ValueError('please calculate center of mass before proceeding')

		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		for snapnum in range(min_snap,max_snap+1):
			snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			pos = snapHDF5.read_block(snapfile, 'POS ', parttype=parttype)/h - CoM_all[snapnum]
			vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=parttype) - v_cm_all[snapnum]
			iD_all = snapHDF5.read_block(snapfile, 'ID ', parttype=parttype)

			vr_allparts = np.array([])
			dist = np.linalg.norm(pos,axis=1)

			for iD in iD_all:
				index = np.where(iD_all == iD)[0][0]
				pos_this = pos[index]
				dist_this = dist[index]
				vel_this = vel[index]

				r_hat_this = pos_this / dist_this
				vr_this = np.dot(r_hat_this,vel_this.T)
				vr_allparts = np.append(vr_allparts,vr_this)

			newfile.create_dataset(str(snapnum).zfill(3),data='vr_allparts')


		#---write--------------------------------------------------------------------------------------
		# if preexist:
		# 	del newfile['SFR_profile_all']; del newfile['drange']

		# newfile.create_dataset('drange',data=drange)
		# newfile.create_dataset('SFR_profile_all',data=SFR_profile_all)
		newfile.close()
		print('\nfinished')
		print('---------------------------------------------------------------')

	newfile.close()

#-----------------------------------------------------------------------------------
def test_mass_profiles():
	'''
	this one writes data after every snapshot so
	it isn't lost if the connection is dropped
	'''
	newfilename = d.datdir+'test_data_logic.hdf5'
	
	print('---------------------------------------------------------------')
	print('testing massprofiles data logic')
	print('writing to: '+str(newfilename))
	print('writing file after every snapshot to preserve data integrity')

	#---open-file-and-figure-out-starting-place----------------------------------------------------
	try:
		newfile = h5py.File(newfilename,'r+')

		try:
			keys = np.array([])
			for k in keysview:
				keys = np.append(keys,k)
			keys.sort()
			min_snap = int(keys[-1][-3:])+1

			print('file exists: opening')
			preexist = True

		except:
			min_snap = 0
			preexist = False
			print('file exists, but is empty. starting from scratch.')
			newfile = h5py.File(newfilename,'w')

	except:
		print('file does not exist: creating')
		min_snap = 0
		preexist = False
		newfile = h5py.File(newfilename,'w')
		
	#---read-snapshot-directory-and-figure-out-ending-place----------------------------------------
	max_snap = 200

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')
		dummy=''

	else:
		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

		for snapnum in np.arange(min_snap,max_snap+1):
			#---print-current-snapshot-------------------------------------------------------------
			snapfile = '/snapshot_'+str(snapnum).zfill(3)
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing)
			sys.stdout.flush()
			if not(snapnum==max_snap):
				sys.stdout.write("\b" * (len(printthing)))
			time.sleep(1)

			#---read-snapshots-and-calculate-------------------------------------------------------
			newdata = np.random.random(100)
			newfile.create_dataset('testdata_'+str(snapnum).zfill(3),data=newdata)
		
		newfile.close()

		print('\nfinished')

	newfile.close()

def testfile(fname):
	# fname = str(input('enter file name: '))
	if not(type(fname)==str): raise ValueError('please input a string.')

	try: f = h5py.File(d.datdir+fname,'r')
	except: raise ValueError('could not find '+d.datdir+fname+'\nplease try again.')

	keysview = f.keys()
	keys = np.array([])
	for k in keysview:
		keys = np.append(keys,k)

	print(d.datdir+fname)
	print('keys and shapes: ')
	for k in keys:
		kd = np.array(f[k])
		print(k+':  '+str(kd.shape))

#-----------------------------------------------------------------------------------