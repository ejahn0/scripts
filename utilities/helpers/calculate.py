import numpy as np
import catalogHDF5 as cat
import snapHDF5


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
	
