import numpy as np
import catalogHDF5 as cat
import directories as d 
import math_helper as m
import snapHDF5, h5py, sys
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

	if not(sim in models):
		raise ValueError('please choose a sim in models')

	i = np.where(models==sim)[0][0]

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
		 
def calculate_sigma_profiles(sim,savename,outdir):#,snapnum=400):
	# snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'

	# if not(sim in models):
	# 	raise ValueError('please choose a sim in models')

	# j = np.where(models==sim)[0][0]

	h=0.7
	all_time = np.array([])
	# drange = np.logspace(-1,2,100)
	print('analyzing simulation: '+sim)

	sigmafile = d.datdir+'sigma_profiles_'+savename+'.hdf5'

	# # sigmafile = h5py.File(sigmaname,'w')

	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(sigmafile))
	drange = np.logspace(-1,np.log10(250),100)

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
	mypath = outdir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

	#---calculate-within-determined-range-of-snapshots-------------------------------------------
	if min_snap>=max_snap:
		print('read all available data. exiting.')

	else:
		print('previously read up to: '+str(min_snap-1)+'\nnow reading up to: '+str(max_snap))

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
			x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
			dark_cm = np.array([x_cm,y_cm,z_cm]).T

			vx_cm = np.sum(dark_vel[:,0] * darkmass) / np.sum(darkmass)
			vy_cm = np.sum(dark_vel[:,1] * darkmass) / np.sum(darkmass)
			vz_cm = np.sum(dark_vel[:,2] * darkmass) / np.sum(darkmass)
			dark_v_cm = np.array([vx_cm,vy_cm,vz_cm]).T

			# d_star = np.linalg.norm(star_pos-dark_cm, axis=1)
			# d_gas  = np.linalg.norm(gas_pos-dark_cm, axis=1)

			star_pos = star_pos-dark_cm
			gas_pos = gas_pos-dark_cm
			star_vel = star_vel-dark_v_cm
			gas_vel = gas_vel-dark_v_cm

			# print(star_pos.shape)

			if len(star_pos) >= 3:
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

def calculate_mass_profiles(sim,savename,outdir):
	# if not(sim in models):
	# 	raise ValueError('please choose a sim in models')

	# i = np.where(models==sim)[0][0]

	newfilename = d.datdir+'massprofiles_'+savename+'.hdf5'
	
	print('---------------------------------------------------------------')
	print('analyzing: '+sim)
	print('writing to: '+str(newfilename))
	drange = np.logspace(-1,np.log10(250),100)

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

	mypath = outdir+sim

	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
	a = np.sort(a)
	max_snap = int(a[-1].split('.')[0][-3:])

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

				if hastype4: 
					type4_profile = np.append(type4_profile,np.sum(type4mass[(d_type4 < dist)]))
				else: 		
					type4_profile = np.append(type4_profile,0.)

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
		#---write--------------------------------------------------------------------------------------
		if preexist:
			del newfile['gas']; del newfile['dark']; del newfile['type2']; del newfile['type3']; del newfile['type4']; del newfile['drange']

		newfile.create_dataset('drange',data=drange)
		newfile.create_dataset('gas',   data=gas_profile_all)
		newfile.create_dataset('dark',  data=dark_profile_all)
		newfile.create_dataset('type2', data=type2_profile_all)
		newfile.create_dataset('type3', data=type3_profile_all)
		newfile.create_dataset('type4', data=type4_profile_all)
		print('\nfinished')

	newfile.close()

	
	print('---------------------------------------------------------------'+'\n')

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

#-----------------------------------------------------------------------------------