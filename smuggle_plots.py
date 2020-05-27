from __future__ import print_function
import os, sys, pdb, h5py, warnings, time, math, snapHDF5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
# from pylab import *
import plot_helper as p
import math_helper as m
import directories as d
import plot_nfw as nfw
import calculate as calc
import mycolors as c
import matplotlib.colors as mplcolors
import matplotlib.cm as cm
print('\n')
h=0.7
warnings.filterwarnings("ignore")

from datetime import datetime
hostname = os.uname()[1]
monthnum = int(str(datetime.now()).split(' ')[0].split('-')[1]) - 1
monthlist = ['01.jan','02.feb','03.mar','04.apr','05.may','06.jun','07.jul','08.aug','09.sep','10.oct','11.nov','12.dec']
month = monthlist[monthnum]
rows, columns = os.popen('stty size', 'r').read().split()
rows = np.int(rows)
columns = np.int(columns)



#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# colors_list =  np.array(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
# 	'#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
# 	'#9a6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#000000'])

# whichsims = 'fixMorph_1e5'
# whichsims = 'fixISM_1e5'
# whichsims = 'fixMorph_1e6'
# whichsims = 'fixISM_1e6'
# whichsims = 'vareff_compare'
# whichsims = 'ff_converge'
# whichsims = '1e5'
# whichsims = '1e6'
whichsims = 'all'
# whichsims = 'comp_cd_rlx_1e5'

#save names should be replaced with models


models = np.array(['fiducial_1e5',					#0
					  'compact_1e5',				#1		cd_fiducial_1e5
					  'tiny_1e5',					#2		cd_ff_tiny_1e5
				   	  'vareff_1e5',					#3
				   	  'vareff_v2_1e5',				#4
				   	  'eSF100_1e5',					#5
				   	  'rho0.1_1e5',					#6
				   	  'fiducial_1e6',				#7
				   	  'eSF100_1e6',					#8
				   	  'rho0.1_1e6',					#9
				   	  'vareff_1e6',					#10
				   	  'tiny_1e6',					#11
				   	  'cd_rlx_1e5',					#12 	cdffrlx_1e5
				   	  'cd_rlx_sf_1e5'				#13])	cdffrlx_sf_1e5
				   	  ])

models_label = np.array(['fiducial 1e5',							#0
						 'compact dwarf 1e5',						#1
						 'tiny dwarf 1e5',							#2
						 'var. eff. 1e5',							#3
						 'var. eff. v2 1e5',						#4
						 r'$\varepsilon_\mathregular{SF}$=1 1e5',	#5
						 r'$n_\mathregular{th}$=0.1cm$^{-3}$ 1e5',	#6
						 'fiducial 1e6',							#7
						 r'$\epsilon_\mathregular{SF}$=1 1e6',		#8
						 r'$n_\mathregular{th}$=0.1cm$^{-3}$ 1e6',	#9
						 'var. eff. 1e6',							#10
						 'tiny dwarf 1e6',							#11
						 'cd adiabatic 1e5',						#12
						 'cd relaxed 1e5'							#13
						 ])

colors_list = np.array(['black',			#0	
				   		'purple',			#1	
				   		'orange',			#2	
				   		'blue',				#3
				   		'darkturquoise',	#4
				   		'firebrick',		#5
				   		'violet',			#6
				   		'grey',			#7
				   		'firebrick',		#8
				   		'violet',			#9
				   		'darkturquoise',	#10
				   		'orange',			#11
				   		'blue',				#12
				   		'red'				#13
				   		])



if whichsims in models:
	mask = np.zeros(len(models)).astype('bool')
	mask[np.where(models==whichsims)[0][0]] = True
														#0      1      2      3      4      5      6      7      8      9      10     11    12    13
elif whichsims=='fixMorph_1e5':			mask = np.array([True , False, False, False, True , True , True , False, False, False, False, False,False,False])
elif whichsims=='fixISM_1e5':			mask = np.array([True , True , True , False, False, False, False, False, False, False, False, False,False,False])
elif whichsims=='fixMorph_1e6':			mask = np.array([False, False, False, False, False, False, False, True , True , True , True , False,False,False])
elif whichsims=='fixISM_1e6':			mask = np.array([False, False, False, False, False, False, False, True , False, False, False, True ,False,False])
elif whichsims=='vareff_compare': 		mask = np.array([False, False, False, True , True , False, False, False, False, False, False, False,False,False])
elif whichsims=='ff_converge':			mask = np.array([True,  False, False, False, False, False, False, True , False, False, False, False,False,False])
elif whichsims=='1e5':					mask = np.array([True , True , True , True , True , True , True , False, False, False, False, False,True ,True ])
elif whichsims=='1e6': 					mask = np.array([False, False, False, False, False, False, False, True , True , True , True , True ,False,False])
elif whichsims=='all':					mask = np.array([True , True , True , True , True , True , True , True , True , True , True , True ,True ,True ])
elif whichsims=='comp_cd_rlx_1e5':		mask = np.array([False, True , False, False, False, False, False, False, False, False, False, False,False,True ])
 
else: raise ValueError('unknown whichsims')

# print('mask',len(mask))
# print('models',len(models))
# print('models_label',len(models_label))
# print('colors_list',len(colors_list))
# print('outdirs',len(outdirs))
# print('savenames',len(savenames))

models = models[mask]
models_label = models_label[mask]
colors_list = colors_list[mask]
outdirs = outdirs[mask]
savenames = savenames[mask]

# savenames = np.array([])
# for model in models:
# 	if 'compact_dwarf' in model:
# 		savenames = np.append(savenames,'cd_'+model.split('/')[-1])
# 	else:
# 		savenames = np.append(savenames,model)


#--------------------------------------------------------------------------------------------------
#---plotting-functions-----------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def testheader():
	num = 69
	sim_name = 'SFlow_1e5'
	fname = d.smuggledir + sim_name + "/snapshot_" + str(num).zfill(3)
	header = snapHDF5.snapshot_header(fname)
	print(header.redshift)

def kslaw():
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	num = 200

	nsim = len(models)
	FileNr =1000

	keep_sfr = np.zeros(shape=(FileNr,nsim))
	keep_gas = np.zeros(shape=(FileNr,nsim))

	for i in range(nsim):
		sim_name = models[i]
		icol = colors_list[i]
		if i == 0:
			isnp0 = 50
			FileNr=200
		if i == 1:
			isnp0 = 50
			FileNr=200
		if i == 2:
			isnp0 = 30
			FileNr= 60

		for isnp in range(isnp0,FileNr+1,10):
			num = isnp

			fname = d.smuggledir + sim_name + "/snapshot_" + str(num).zfill(3) #"%03d" % (num) 
			print(fname)

			header = snapHDF5.snapshot_header(fname)
			Ngas = header.nall[0]
			print("Num of gas=%s" % Ngas)

			rho = snapHDF5.read_block(fname,"RHO ",parttype=0)
			nHI = snapHDF5.read_block(fname,"NH  ",parttype=0)
			sfr = snapHDF5.read_block(fname,"SFR ",parttype=0)
			U = snapHDF5.read_block(fname,"U   ", parttype=0)
			Nelec = snapHDF5.read_block(fname,"NE  ",parttype=0)

			rho *= m.Xh/m.PROTONMASS*m.UnitDensity_in_cgs  #now in cm^{-3} h^3
			MeanWeight= 4.0/(3*m.Xh+1+4*m.Xh*Nelec) * m.PROTONMASS
			temp = MeanWeight/m.BOLTZMANN * (m.gamma-1) * U * m.UnitEnergy_in_cgs/ m.UnitMass_in_g
			
			sfr *= 1.e10/1.e9  #Msun/yr, check!!

			pos_gas = snapHDF5.read_block(fname,"POS ",parttype=0)
			pos_str = snapHDF5.read_block(fname,"POS ",parttype=4)
			mass_str = snapHDF5.read_block(fname,"MASS",parttype=4)
			mstr_tot = np.sum(mass_str)

			if mstr_tot > 0:
				x0 = np.sum(pos_str[:,0]*mass_str[:]) / mstr_tot
				y0 = np.sum(pos_str[:,1]*mass_str[:]) / mstr_tot
				z0 = np.sum(pos_str[:,2]*mass_str[:]) / mstr_tot
			else:
				x0 = 100.
				y0 = x0
				z0 = x0

			pos_gas[:,0] -= x0
			pos_gas[:,1] -= y0
			pos_gas[:,2] -= z0
			# w = pos_gas[:,0]**2 + pos_gas[:,1]**2 + pos_gas[:,2]**2
			# rgas = np.sqrt(w)
			rgas = np.linalg.norm(pos_gas-np.array([0.,0.,0.]), axis=1)
			mgas = snapHDF5.read_block(fname,"MASS",parttype=0)

			#ix=rgas < 20.1
			#ix = np.argwhere((rgas < 20.1) & (abs(pos_gas[:,2]) < 1.)).flatten()   ##not real 2D yet, but quite close
			ix = np.argwhere((rgas < 15.1)).flatten()   ##not real 2D yet, but quite close
			rgas = rgas[ix]
			mgas = mgas[ix]
			sfr  = sfr[ix]
			ind = np.argsort(rgas)
			rgas = rgas[ind]
			mgas = mgas[ind]
			sfr  = sfr[ind]

			sfr_acum = np.cumsum(sfr)
			sfr_half = sfr_acum[len(sfr_acum)-1]/2.
			iix = sfr_acum > sfr_half
			rr = rgas[iix]
			rh = rr[0]

			iix2 = rgas < rh
			mgas_rh = np.sum(mgas[iix2])
			sigma_gas = mgas_rh * 1.e10 / (3.14159 * (rh*1000.)**2)   #pc^-2
			sigma_sfr = sfr_half /  (3.14159 * (rh)**2)  #kpc^-2

			keep_sfr[isnp,i] = sigma_sfr # * HubbleParam
			keep_gas[isnp,i] = sigma_gas #* HubbleParam

			## plot
			kk = np.argwhere((keep_gas[:,i] > 0) & (keep_sfr[:,i] > 0)).flatten()
			xx = np.log10(keep_gas[kk,i])
			yy = np.log10(keep_sfr[kk,i])
			ax.plot(xx,yy,'o',markersize=8,color=icol,alpha=0.5,mew=0)  # connect points with a blue line

	# plt.axis([-0.4,4.6,-3.,3.0])

	xx1 = [0.,4.6]
	yy1 = [-2.4,3.9]
	xx2 = [1.05,4.6]
	yy2 = [-2.4,2.1]
	ax.plot(np.array(xx1),np.array(yy1),":",linewidth=3,color='black')
	ax.plot(np.array(xx2),np.array(yy2),":", linewidth=3,color='black')

	ax.set_xlabel(r'${\bf\rm log(\Sigma_{\rm gas}) [\rm M_\odot pc^{-2}]}$')
	ax.set_ylabel(r'${\bf\rm log(\Sigma_{\rm SFR}) [\rm M_\odot yr^{-1} kpc^{-2}]}$')

	## LABELS ##
	# ax = fig.add_axes([0.,0.,1.,1.])
	# ax.set_axis_off()
	ax.set_xlim(0.5,2)
	ax.set_xticks([1,2])
	ax.set_xticks([0.5,1.5],minor=True)
	ax.set_ylim(-2.5,1.5)
	ax.set_yticks([-2,-1,0,1])
	ax.set_yticks([-1.5,-0.5,0.5],minor=True)

	# x0 = 0.15
	# y0 = 0.85
	for i in range(nsim):
		# icol = colors_list[i]
		# text = models_label[i]
		# ax.annotate(x0,y0,text,color=icol,fontsize=12)
		ax.annotate(models_label[i],xy=(0.05,1-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])
		# y0 -= 0.05

	p.finalize(fname='kslaw_allmodels',save=1)

def sfr_time(do_bins=False):
	fig,ax=p.makefig(1,figx=8,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for sim in models:
		# print(np.where(models==sim))
		i = np.where(models==sim)[0][0]
		filein = d.smuggledir + sim + '/sfr.txt'
		print(filein)

		try:
			data = np.loadtxt(filein)
		except:
			print('np.loadtxt failed, reading manually')
			f = open(filein,'r')
			j = 0

			for line in f:
				linearray = np.array(line.split(' '))
				linearray = linearray[np.where(np.logical_not(linearray==''))]
				if j==0:
					data = linearray
				else:
					data = np.vstack((data,linearray))
				j += 1
			f.close()

		
		time = data[:,0]
		sfr = data[:,2]

		kk = sfr == 0
		sfr[kk]=1.e-10

		if do_bins:
			timebins = np.linspace(0, 2, 200)
			binwidth = (timebins[1] - timebins[0])/2
			sfr_mean = np.array([])

			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				sel = (time > leftbin) & (time < rightbin)
				sfr_mean = np.append(sfr_mean, np.mean(sfr[sel]))

			ax.plot(timebins[0:-1]+binwidth,sfr_mean,color=colors_list[i],linewidth=1.6,alpha=0.7,ls='-')
			ax.annotate(models_label[i],xy=(0.05,1-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])

		else:
			ax.plot(time,sfr,color=colors_list[i],linewidth=1.3,alpha=0.9,ls='-')
			ax.annotate(models_label[i],xy=(0.05,1-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])

	
	ax.set_xlim([0,2])
	# ax.set_xlim([0.8,1.2])
	ax.set_xticks([0,0.5,1,1.5,2])
	ax.set_xticks([0.25,0.75,1.25,1.75],minor=True)
	ax.set_xlabel(r'time [$h^{-1}$Gyr]')

	ax.set_yscale('log')
	ax.set_ylim([1.e-3,1e1])
	ax.set_ylabel(r'SFR [M$_\odot$/yr]')
	
	p.finalize(fig,fname='sfrtime_'+whichsims,save=1)#,tight=True)

def sfr_time_compare(do_bins=False):
	fig,ax=p.makefig(1,figx=8,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	nmod = len(models)
	res = '1e5'
	colors = np.array(['black','purple','green'])
	olddir = '/mainvol/ejahn/smuggle/output/live_halo_09.2019/'
	newdir = '/mainvol/ejahn/smuggle/output/live_halo_02.2020/'

	for i in range(nmod):
		oldin = olddir + models[i] + '/sfr.txt'
		newin = newdir + models[i] + '/sfr.txt'
		
		print('old filein=',oldin)
		print('new filein=',newin)

		old_data = np.loadtxt(oldin)
		old_time = old_data[:,0]
		old_sfr = old_data[:,2]
		old_sfr[(old_sfr == 0)]=1.e-10

		new_data = np.loadtxt(newin)
		new_time = new_data[:,0]
		new_sfr = new_data[:,2]
		new_sfr[(new_sfr == 0)]=1.e-10

		if do_bins:
			timebins = np.linspace(0, 1, 100)
			binwidth = (timebins[1] - timebins[0])/2
			sfr_mean = np.array([])

			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				sel = (time > leftbin) & (time < rightbin)
				sfr_mean = np.append(sfr_mean, np.mean(sfr[sel]))

			ax.plot(timebins[0:-1]+binwidth,sfr_mean,color=colors[i],linewidth=2,alpha=0.9,ls='-')

		else:
			ax.plot(old_time,old_sfr,color=colors[i],linewidth=2,alpha=0.9,ls='-')
			ax.plot(new_time,new_sfr,color=colors[i],linewidth=1.3,alpha=0.9,ls='--')

	ax.set_yscale('log')
	ax.set_ylim([1.e-3,1e1])
	ax.set_xlim([0,1.])
	ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
	ax.set_xticks([0.1,0.3,0.5,0.7,0.9],minor=True)
	ax.set_xlabel(r'time [$h^{-1}$Gyr]')
	ax.set_ylabel(r'SFR [M$_\odot$/yr]')

	x0 = 0.25
	y0 = 0.85
	for i in range(nmod):
		ax.annotate(models_label[i],xy=(0.05,1-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors[i])

	p.finalize(fig,fname='sfrtime_newold',save=1,tight=True)

def xyz_stars():
	import snapHDF5_mine as snap
	figname ='sfr_fromstars_magenta_ffnr.png'
	num = 199
	sim_name = models[0]

	fig = plt.figure()
	ax = plt.subplot(111)
	nmod = len(models)
	fig.set_size_inches(6.5,6.5)

	fname = d.smuggledir + sim_name + "/snapshot_" +  "%03d" % (num) ## + ".hdf5"
	print(fname)

	header = snap.snapshot_header(fname)
	Ngas = header.nall[0]
	print("Num of gas=%s" % Ngas)

	pos_str = snap.read_block(fname,"POS ",parttype=4)
	mass_str = snap.read_block(fname,"MASS",parttype=4)
	tform = snap.read_block(fname,"GAGE",parttype=4)
	rstrom = snap.read_block(fname,"STRM",parttype=4)
	mstr_tot = np.sum(mass_str)
	mass_init = snap.read_block(fname,"GIMA",parttype=4)
	if mstr_tot > 0:
		x0 = np.sum(pos_str[:,0]*mass_str[:]) / mstr_tot
		y0 = np.sum(pos_str[:,1]*mass_str[:]) / mstr_tot
		z0 = np.sum(pos_str[:,2]*mass_str[:]) / mstr_tot
	else:
		x0 = 100.
		y0 = x0
		z0 = x0

	pos_str[:,0] -= x0
	pos_str[:,1] -= y0
	pos_str[:,2] -= z0

	dist_str = np.sqrt(pos_str[:,0]**2 + pos_str[:,1]**2 + pos_str[:,2]**2)

	print('number of stars=',len(pos_str[:,0]))
	print('max distance stars',np.max(dist_str))

	#plt.plot(pos_str[:,0],pos_str[:,1],'bo')
	plt.plot(pos_str[:,0],pos_str[:,1],'ro')
	## --- gas

	pos_gas = snap.read_block(fname,"POS ",parttype=0)
	pos_gas[:,0] -= x0
	pos_gas[:,1] -= y0
	pos_gas[:,2] -= z0

	plt.plot(pos_gas[:,0],pos_gas[:,1],'b.',markersize=1.1,alpha=0.4)

	rr = 5
	plt.axis([-rr,rr,-rr,rr])


	plt.savefig(figname, format='png')

def rho_temp(sim,num):
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	figname = 'rho_temp_'+sim

	fname = d.smuggledir + sim + "/snapshot_" + str(num).zfill(3)
	print(fname)

	header = snapHDF5.snapshot_header(fname)
	Ngas = header.nall[0] 
	print("Num of gas=%s" % Ngas)

	rho = snapHDF5.read_block(fname,"RHO ",parttype=0)
	rho *= m.Xh/m.PROTONMASS*m.UnitDensity_in_cgs  #now in cm^{-3} h^3 
	U = snapHDF5.read_block(fname,"U   ",parttype=0)
	Nelec = snapHDF5.read_block(fname,"NE  ",parttype=0)
	MeanWeight= 4.0/(3*m.Xh+1+4*m.Xh*Nelec) * m.PROTONMASS
	temp = MeanWeight/m.BOLTZMANN * (m.gamma-1) * U * m.UnitEnergy_in_cgs/ m.UnitMass_in_g
	nHI = snapHDF5.read_block(fname,"NH  ",parttype=0) 

	sfr = snapHDF5.read_block(fname,"SFR ",parttype=0)
	pos_gas = snapHDF5.read_block(fname,"POS ",parttype=0)
	pos_str = snapHDF5.read_block(fname,"POS ",parttype=4)
	mass_str = snapHDF5.read_block(fname,"MASS",parttype=4)
	mstr_tot = np.sum(mass_str)
	x0 = np.sum(pos_str[:,0]*mass_str[:]) / mstr_tot
	y0 = np.sum(pos_str[:,1]*mass_str[:]) / mstr_tot
	z0 = np.sum(pos_str[:,2]*mass_str[:]) / mstr_tot
	print('pos CM=',x0,y0,z0)
	   ### --- some notes ---
	   #pos_gas.shape  =(Ngas,3)
	   #pos_gas.size  =Ngas*3
	   #pos_gas[:,0].size = Ngas
	   ##-------------
	pos_gas[:,0] -= x0 
	pos_gas[:,1] -= y0 
	pos_gas[:,2] -= z0 
	w = pos_gas[:,0]**2 + pos_gas[:,1]**2 + pos_gas[:,2]**2
	rgas = np.sqrt(w)

	rlim = 3. #20.
	ix=rgas < rlim 
	#ix=rgas < 5.1
	print("length inner radius: ",len(ix))
	kk = np.where(rgas <rlim) 
	kk_sfr = (rgas <rlim) & (sfr>0.) 

	## -- plot
	#plt.plot(rho,temp,'r.')  # connect points with a blue line
	ax.loglog(rho[kk],temp[kk],'b.',markersize=2.)  # connect points with a blue line
	ax.loglog(rho[kk_sfr],temp[kk_sfr],'r.',markersize=2.)  # connect points with a blue line
	# ax.set_xlabel(r'${\bf\rm Density [\rm cm^{-3}]}$',fontsize=13) 
	# ax.set_ylabel(r'${\bf\rm Temperature [^\circ K]}$',fontsize=13) 
	ax.set_xlabel(r'Density [cm$^\mathregular{-3}$]')
	ax.set_ylabel(r'Temperature [K]')
	#plt.xlim([1.e-4,1.e4])
	ax.set_xlim([1.e-4,1.e4])
	ax.set_ylim([10,1.e7])
	# ax.set_title(sim+'   ',loc='right',size=10)
	simtext = models_label[np.where(models==sim)[0][0]]
	ax.annotate(simtext,xy=(0.7,0.9),xycoords='axes fraction',fontsize=15,color='k')

	r1 = rgas[kk]
	rho1 = rho[kk]
	temp1 = temp[kk]
	kk2 = (rho1>1.) & (temp1>1.5e4)

	f = float(len(rho1[kk2]))/float(len(rho1))
	print('fraction ionized=',f)
	#plt.loglog(nHI[kk],temp[kk],'b.',markersize=0.3)  # connect points with a blue line

	p.finalize(fig,fname=figname,save=1)

def density_profile(snapnum):
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	snapnum = str(snapnum).zfill(3)
	snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'
	h = 0.7
	lss = np.array(['-','--'])

	# models = np.array(['SFhigh_1e5'])

	#-read-snapshot-data---------------------------------------------------
	for model in models:
		i = np.where(models==model)[0][0]
		snapfile = snapdir+model+'/snapshot_'+snapnum
		# starmass = snap.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
		# star_pos = snap.read_block(snapfile, 'POS ', parttype=4)/h
		# gasmass  = snap.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
		# gas_pos  = snap.read_block(snapfile, 'POS ', parttype=0)/h
		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
		dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

		#-calculate-center-of-mass---------------------------------------------
		x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
		dark_cm = np.array([x_cm,y_cm,z_cm]).T
		med_dark_pos = np.array([np.median(dark_pos[:,0]),np.median(dark_pos[:,1]),np.median(dark_pos[:,2])])
		# med_star_pos = np.array([np.median(star_pos[:,0]),np.median(star_pos[:,1]),np.median(star_pos[:,2])])
		# med_gas_pos = np.array([np.median(gas_pos[:,0]),np.median(gas_pos[:,1]),np.median(gas_pos[:,2])])

		host_pos = dark_cm
		# d_str = np.linalg.norm(star_pos-host_pos, axis=1)
		# d_gas = np.linalg.norm( gas_pos-host_pos, axis=1)
		d_dark = np.linalg.norm(dark_pos-host_pos, axis=1)

		# drange = np.arange(0,100,20)
		drange = np.logspace(-1,2,200)
		rho_dm = np.array([]); rho_str = np.array([]); rho_gas = np.array([])

		for d in drange:
			vol = (4./3.)*np.pi*d**3
			sel_dm_in_d = (d_dark < d)
			rho_dm = np.append(rho_dm, np.sum(darkmass[sel_dm_in_d])/vol)
			# sel_str_in_d = (d_str < d)
			# rho_str = np.append(rho_str, np.sum(starmass[sel_str_in_d])/vol)
			# sel_gas_in_d = (d_gas < d)
			# rho_gas = np.append(rho_gas, np.sum(gasmass[sel_gas_in_d])/vol)

		ax.plot(drange,rho_dm,color=colors_list[i],label=models_label[i],lw=2,alpha=0.7)
		ax.annotate('snapshot '+snapnum,xy=(0.1,0.1),xycoords='axes fraction', fontsize=12,color='k')
		# xycoords='axes fraction', fontsize=12,color=colors[i]
		ax.annotate(models_label[i],xy=(0.55,1.-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])

	#-plot--------------------------------------------------------------------
	# ax.legend(frameon=False,loc='lower left',prop={'size':11})
		
	ax.set_xscale('log')
	ax.set_xlim(1e-1,5e0)
	ax.set_xlabel('distance [kpc]')
	ax.set_xticks([0.1,0.2,0.5,1,2,5])
	ax.set_xticklabels([0.1,0.2,0.5,1,2,5])

	ax.set_yscale('log')
	ax.set_ylim(2e7,7e8)
	ax.set_ylabel(r'density [M$_\odot$ kpc$^\mathregular{-3}$]')

	p.finalize(fname='density_profile_all',save=1)

def multisnap_density_prof():
	snapnums = np.array(['050','100','150','200'])
	snapdir = '/home/ejahn003/smuggle/dwarf/iso/live_halo/out/'
	h = 0.7

	fig = plt.figure()
	ax = plt.subplot(111)
	# ax.text(2e-1,1e-2,'snapshot '+snapnum)

	colors = np.array(['blue','green','orange','red'])
	lss = np.array(['-','--'])

	# i=0 #snapnum index
	j=0 #model index

	for model in models:
		for snapnum in snapnums:
			i = np.where(snapnums==snapnum)[0][0]
			print('calculating '+model+' snapshot '+snapnum+'; i='+str(i))
			snapfile = snapdir+model+'/snapshot_'+snapnum
			darkmass = snap.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
			dark_pos = snap.read_block(snapfile, 'POS ', parttype=1)/h

			#-calculate-center-of-mass---------------------------------------------
			x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
			y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
			z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
			dark_cm = np.array([x_cm,y_cm,z_cm]).T
			med_dark_pos = np.array([np.median(dark_pos[:,0]),np.median(dark_pos[:,1]),np.median(dark_pos[:,2])])
			d_dark = np.linalg.norm(dark_pos-dark_cm, axis=1)

			drange = np.logspace(-1,1,200)
			rho_dm = np.array([])

			for d in drange:
				vol = (4./3.)*np.pi*d**3
				sel_dm_in_d = (d_dark < d)
				rho_dm = np.append(rho_dm, np.sum(darkmass[sel_dm_in_d])/vol)

			#-plot-the-stuff----------------------------------------------------------
			ax.plot(drange,rho_dm,c=colors[i],ls=lss[j],lw=2)
			if j==0:
				ax.annotate('snapshot '+snapnum, xy=(0.75, 1.-(0.05*(i+4))), xycoords='axes fraction', fontsize=12,color=colors[i])
		j+=1

	ax.plot(1e10,1e10,'-',c='k',label='full feedback')
	ax.plot(1e10,1e10,'--',c='k',label='no radiation')
	ax.legend(frameon=False, prop={'size':12})
	ax.set_xlabel('distance [kpc]',size=12)
	ax.set_ylabel(r'density [M$_\odot$ kpc$^\mathregular{-3}$]',size=12)

	ax.set_xscale('log')
	ax.set_yscale('log')

	ax.set_ylim(1e8,1e9)
	ax.set_xlim(1e-1,1e0)

	ax.set_xticks([0.1,0.2,0.4,0.7,1])
	ax.set_xticklabels([0.1,0.2,0.4,0.7,1])

	plt.savefig('multisnap_density_profile.png',format='png',dpi=300)

def panel_projection_single(sim,snapnum,bound=10,show_progress=True,do_rhalf=False,do_rcore=False,do_scale=False):

	if sim in models:
		i = np.where(models==sim)[0][0]
		thislabel = models_label[i]
	else:
		raise ValueError('please choose a sim in models')

	savedir = '/home/ejahn003/movie_frames/'+savenames[i]+'/'
	savedir = '/home/ejahn003/plots/05.may/'+savenames[i]

	fname = outdirs[i]+sim+'/snapshot_' + str(snapnum).zfill(3)

	# fig, axarr = p.makefig(n_panels='2_proj')
	fig, axarr = p.makefig(n_panels=2)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	plt.style.use('dark_background')

	#---------------------------------------------
	gas_pos = snapHDF5.read_block(fname,"POS ",parttype=0)
	gasmass = snapHDF5.read_block(fname,"MASS", parttype=0)*(1.e10)/m.h
	str_pos = snapHDF5.read_block(fname,"POS ",parttype=4)
	strmass = snapHDF5.read_block(fname,"MASS", parttype=4)*(1.e10)/m.h
	darkpos = snapHDF5.read_block(fname,"POS ",parttype=1)
	darkmass = snapHDF5.read_block(fname,"MASS", parttype=1)*(1.e10)/m.h

	#---------------------------------------------
	x_cm = np.sum(darkpos[:,0] * darkmass) / np.sum(darkmass)
	y_cm = np.sum(darkpos[:,1] * darkmass) / np.sum(darkmass)
	z_cm = np.sum(darkpos[:,2] * darkmass) / np.sum(darkmass)
	cm = np.array([x_cm,y_cm,z_cm]).T

	darkpos = darkpos - cm
	str_pos = str_pos - cm
	gas_pos = gas_pos - cm

	#---------------------------------------------
	gas_ap = 0.5
	star_ap = 1

	#---plot-face-on---
	sel_gas_front = (gas_pos[:,2] > 0)
	sel_gas_back = (gas_pos[:,2] <= 0)
	
	axarr[0].plot(gas_pos[:,0][sel_gas_front],gas_pos[:,1][sel_gas_front],',',c='blue',alpha=gas_ap,zorder=100)
	if np.sum(strmass) > 0:
		axarr[0].plot(str_pos[:,0],str_pos[:,1],',',c='gold',alpha=star_ap,zorder=10)
	axarr[0].plot(gas_pos[:,0][sel_gas_back],gas_pos[:,1][sel_gas_back],',',c='blue',alpha=gas_ap,zorder=1)	
	
	#---plot-edge-on---
	sel_gas_front = (gas_pos[:,1] > 0)
	sel_gas_back = (gas_pos[:,1] <= 0)	

	axarr[1].plot(gas_pos[:,0][sel_gas_front],gas_pos[:,2][sel_gas_front],',',c='blue',alpha=0.2,zorder=100)	
	if np.sum(strmass) > 0:
		axarr[1].plot(str_pos[:,0],str_pos[:,2],',',c='gold',alpha=star_ap,zorder=10)
	axarr[1].plot(gas_pos[:,0][sel_gas_back],gas_pos[:,2][sel_gas_back],',',c='blue',alpha=gas_ap,zorder=1)

	height = 0.88

	if do_rhalf:
		f = h5py.File(d.datdir+'massprofiles_'+savenames[i]+'.hdf5','r')
		drange = np.array(f['drange'])
		star_profile = np.array(f['type4'])[snapnum]
		f.close()
		total_star_mass = np.sum(strmass)
		if total_star_mass > 0:
			fractional_profile = star_profile / total_star_mass
			half_index = np.argmin(np.abs(fractional_profile-0.5))
			star_radius = drange[half_index]
			circle1 = plt.Circle((0, 0), star_radius, ec='lime',fc='none',lw=1.5,zorder=1e4)
			axarr[0].add_artist(circle1)
			circle2 = plt.Circle((0, 0), star_radius, ec='lime',fc='none',lw=1.5,zorder=1e4)
			axarr[1].add_artist(circle2)

			axarr[0].annotate(r'$r_{\mathregular{h}\ast}$', xy=(0.05, height), xycoords='axes fraction', fontsize=12,color='lime')
			height = height-0.05

	if do_rcore:
		f = h5py.File(d.datdir+'massprofiles_'+savenames[i]+'.hdf5','r')
		drange = np.array(f['drange'])
		dark_profile_all = np.array(f['dark'])
		f.close()
		dark_profile = dark_profile_all[snapnum]
		dark_profile_0 = dark_profile_all[0]
		vols = 4./3.*np.pi*(drange**3)
		density = dark_profile/vols
		density_0 = dark_profile_0/vols
		
		rho_ratio = density_0 / density
		sel_in_2 = (rho_ratio > 1.4) & (rho_ratio < 1.8)
		is_2 = np.count_nonzero(sel_in_2)

		if is_2 > 0:
			ind = np.where(sel_in_2)[0][-1]
			core_radius = drange[ind]
			circle3 = plt.Circle((0, 0), core_radius, ec='fuchsia',fc='none',lw=1.5,zorder=1e4)
			axarr[0].add_artist(circle3)
			circle4 = plt.Circle((0, 0), core_radius, ec='fuchsia',fc='none',lw=1.5,zorder=1e4)
			axarr[1].add_artist(circle4)

			axarr[0].annotate(r'$r_\mathregular{core}$', xy=(0.05, height), xycoords='axes fraction', fontsize=12,color='fuchsia')

	if do_scale:
		sl = bound/5
		ys = -0.9*bound
		xs = -0.9*bound
		axarr[0].plot(np.array([xs,xs+sl]),np.array([ys,ys]),c='red',lw=2,alpha=0.6)
		axarr[0].annotate(str(np.round(sl,1))+' kpc', xy=(-0.9*bound, -0.85*bound), fontsize=12,color='red',alpha=0.6)


	#---------------------------------------------
	axarr[0].set_xlim(-bound,bound)
	axarr[0].set_ylim(-bound,bound)

	axarr[1].set_xlim(-bound,bound)
	axarr[1].set_ylim(-bound,bound)
	
	axarr[0].annotate(thislabel, xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12,color='white')

	if show_progress:
		header = snapHDF5.snapshot_header(fname)
		time = np.round(header.time,1)

		px = (np.float(snapnum)/400. * 20. - 10.)*(18./20.)
		pxs = np.array([-9.,px])
		pys = np.array([-9.,-9.])

		axarr[1].plot(pxs,pys,'-',c='red',lw=3,alpha=0.4)
		axarr[1].annotate(r'$t$ = '+str(time)+' Gyr', xy=(-9.3,-8.1),fontsize=10,color='red',alpha=0.6)
		axarr[1].annotate('[',xy=(-9.3,-9.1),color='red',fontsize=10,alpha=0.4)
		axarr[1].annotate(']',xy=(9.2,-9.1),color='red',fontsize=10,alpha=0.4)

	figname = savedir#+'snapshot_'+str(snapnum).zfill(3)#+'.png'

	if do_rhalf:
		figname = figname + '_rhalf'
	if do_rcore:
		figname = figname + '_rcore'
	figname = figname + '_snapshot_'+str(snapnum).zfill(3)+'.png'

	print('saving figure: '+figname)
	plt.savefig(figname,format='png',dpi=200)
	# plt.show()

def stellar_mass_growth():
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for sim in models:
		i = np.where(models == sim)[0][0]
		print(sim)

		mstr_of_t = np.array([])
		all_time = np.array([])

		for n in np.arange(201):
			fname = d.smuggledir + sim + '/snapshot_' + str(n).zfill(3)
			printthing = 'calculating snapshot '+str(n).zfill(3)+'/200'
			sys.stdout.write(printthing)
			sys.stdout.flush()
			sys.stdout.write("\b" * (len(printthing)))

			strmass = snapHDF5.read_block(fname, 'MASS', parttype=4)*(1.e10)/m.h
			mstr_of_t = np.append(mstr_of_t, np.sum(strmass))

			header = snapHDF5.snapshot_header(fname)
			all_time = np.append(all_time, header.time)

		print('\n')
		ax.plot(all_time,mstr_of_t,'-',c=colors_list[i],label=models_label[i],lw=2,alpha=0.6)
		ax.annotate(models_label[i],xy=(0.05,1.-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])

	ax.set_xlabel(r'time [$h^{-1}$ Gyr]')
	ax.set_xscale('log')
	ax.set_xlim(1e-2,1e0)
	# ax.set_xticks([5,10,20,50,100,200])
	# ax.set_xticklabels([5,10,20,50,100,200])

	ax.set_ylabel(r'M$_\mathregular{star}$ [M$_\odot$]')
	ax.set_yscale('log')
	
	p.finalize(fname='stellar_mass_growth_all',save=1)
	
def density_sfr(model,snapnum,create_file=False):
	fig, axarr = p.makefig('density thing')

	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	snapnum = str(snapnum).zfill(3)
	snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'
	h = 0.7
	# model = 'fiducial_1e6'
	# drange = np.logspace(-1,2,100)
	i = np.where(models==model)[0][0]

	#----calculate-density-profile--------------------------------------------------------------------------
	#-------------------------------------------------------------------------------------------------------
	snapfile = snapdir+model+'/snapshot_'+snapnum
	darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
	dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
	header = snapHDF5.snapshot_header(snapfile)
	rho_time = header.time

	if create_file:
		rhofile = h5py.File(d.datdir+'/rhodm_'+model+'.hdf5','w')

		x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
		dark_cm = np.array([x_cm,y_cm,z_cm]).T
		d_dark = np.linalg.norm(dark_pos-dark_cm, axis=1)

		rho_dm = np.array([])
		for r in drange:
			vol = (4./3.)*np.pi*r**3
			sel_dm_in_d = (d_dark < r)
			rho_dm = np.append(rho_dm, np.sum(darkmass[sel_dm_in_d])/vol)

		rhofile.create_dataset('snap_'+str(snapnum).zfill(3),data=rho_dm)

	else:
		rhofile = h5py.File(d.datdir+'/rhodm_'+model+'.hdf5','r')
		rho_dm = np.array(rhofile['snap_'+str(snapnum).zfill(3)])
		drange = np.array(rhofile['drange'])
		rhofile.close()

	axarr[0].plot(drange,rho_dm,color='b',lw=2,alpha=0.7)
	axarr[0].annotate('t = '+str(int(rho_time*1e3))+' Myr',xy=(0.1,0.1),xycoords='axes fraction', fontsize=12,color='k')
	axarr[0].annotate(models_label[i],xy=(0.1,0.15),xycoords='axes fraction', fontsize=12,color='k')


	#-plot-----------------------------------------------------------------------------------------------------
	axarr[0].set_xscale('log')
	axarr[0].set_xlim(1e-1,5e0)
	axarr[0].set_xlabel('distance [kpc]')
	axarr[0].set_xticks([0.1,0.2,0.5,1,2,5])
	axarr[0].set_xticklabels([0.1,0.2,0.5,1,2,5])

	axarr[0].set_yscale('log')
	axarr[0].set_ylim(2e7,7e8)
	axarr[0].set_ylabel(r'density [M$_\odot$ kpc$^\mathregular{-3}$]')

	#-------------------------------------------------------------------------------------------------------
	#-------------------------------------------------------------------------------------------------------
	filein = d.smuggledir + model + '/sfr.txt'
	data = np.loadtxt(filein)
	time = data[:,0]
	sfr = data[:,2]

	kk = sfr == 0
	sfr[kk]=1.e-10
	axarr[1].plot(time,sfr,color='b',linewidth=1.3,alpha=0.9)
	axarr[1].axvline(x=rho_time,color='red',lw=2,ls='-',alpha=0.6)

	axarr[1].set_yscale('log')
	axarr[1].set_ylim([1.e-4,30])
	axarr[1].set_xlim([0,1.])
	axarr[1].set_xlabel(r'time [$h^{-1}$Gyr]')
	axarr[1].set_ylabel(r'SFR [M$_\odot$/yr]')

	#-------------------------------------------------------------------------------------------------------
	#-------------------------------------------------------------------------------------------------------
	# p.finalize(save=False)
	this_dir = '/home/ejahn003/plots/'+month+'/rho_sfr_'+model+'/'

	print('saving '+model+' #'+snapnum)

	plt.savefig(this_dir+snapnum+'.png',format='png',dpi=100)

def make_movie_frames(sims):
	snaplist = np.arange(9,201)
	for num in snaplist:
		print(num)
		for sim in sims:
			panel_projection_single(sim,num,'1e5')

def plot_mass_in_r(sim):
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	R_measures = np.array([0.1,0.2,0.5,1.,2.,5])

	f = h5py.File(d.datdir+'mass_in_r_'+sim+'.hdf5','r')
	darkmasses_in_Rs = np.array(f['darkmasses_in_Rs'])
	gasmasses_in_Rs = np.array(f['gasmasses_in_Rs'])
	time = np.array(f['time'])
	f.close()

	mdark_in_01 = darkmasses_in_Rs[:,0]
	mdark_in_02 = darkmasses_in_Rs[:,1]
	mdark_in_05 = darkmasses_in_Rs[:,2]
	mdark_in_1 = darkmasses_in_Rs[:,3]
	mdark_in_2 = darkmasses_in_Rs[:,4]
	mdark_in_5 = darkmasses_in_Rs[:,5]

	mgas_in_01 = gasmasses_in_Rs[:,0]
	mgas_in_02 = gasmasses_in_Rs[:,1]
	mgas_in_05 = gasmasses_in_Rs[:,2]
	mgas_in_1 = gasmasses_in_Rs[:,3]
	mgas_in_2 = gasmasses_in_Rs[:,4]
	mgas_in_5 = gasmasses_in_Rs[:,5]

	#---------------------------------------------------------------
	timebins = np.linspace(0, 1, 50)
	binwidth = (timebins[1] - timebins[0])/2

	mg01_mean = np.array([])
	mg02_mean = np.array([])
	mg05_mean = np.array([])
	mg1_mean  = np.array([])
	mg2_mean  = np.array([])
	mg5_mean  = np.array([])

	for i in range(len(timebins)-1):
		leftbin = timebins[i]
		rightbin = timebins[i+1]

		sel = (time > leftbin) & (time < rightbin)
		mg01_mean = np.append(mg01_mean, np.mean(mgas_in_01[sel]))
		mg02_mean = np.append(mg02_mean, np.mean(mgas_in_02[sel]))
		mg05_mean = np.append(mg05_mean, np.mean(mgas_in_05[sel]))
		mg1_mean  = np.append(mg1_mean , np.mean(mgas_in_1[sel]))
		mg2_mean  = np.append(mg2_mean , np.mean(mgas_in_2[sel]))
		mg5_mean  = np.append(mg5_mean , np.mean(mgas_in_5[sel]))
	#----------------------------------------------------------------------------------------------
	#----------------------------------------------------------------------------------------------
	#----------------------------------------------------------------------------------------------

	f = h5py.File(d.datdir+'notcusp_radius.hdf5','r')
	# corefile.create_dataset(sim+'_coreradius',data=all_core_radius)
	# corefile.create_dataset(sim+'_coreradius_smooth',data=coreradius_mean)
	# corefile.create_dataset(sim+'_alltime',data=all_time)
	# corefile.create_dataset(sim+'_time_smooth',data=timebins[0:-1]+binwidth)

	r_core = np.array(f[sim+'_coreradius_smooth'])
	rctime = np.array(f[sim+'_time_smooth'])
	f.close()

	rc_max = np.amax(r_core)
	sel = (r_core > 0.75*rc_max)
	shadetimes = rctime[sel]

	y0 = np.zeros(len(rctime))
	y1 = np.zeros(len(rctime))+1e5
	y2 = np.zeros(len(rctime))+1e10

	ax2 = ax.twinx()

	ax2.fill_between(rctime,y0,r_core,facecolor='orange', alpha=0.15, zorder=0, edgecolor='none')
	ax2.set_ylim(0,0.7)
	ax2.set_ylabel('core radius [kpc]')
	ax2.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
	ax2.set_yticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65],minor=True)

	#----------------------------------------------------------------------------------------------
	#----------------------------------------------------------------------------------------------
	#----------------------------------------------------------------------------------------------
	ax.plot(time,mdark_in_01,'-',color='red'   ,zorder=10)
	ax.plot(time,mdark_in_02,'-',color='orange',zorder=10)
	ax.plot(time,mdark_in_05,'-',color='green' ,zorder=10)
	ax.plot(time,mdark_in_1,'-',color= 'blue'  ,zorder=10)
	ax.plot(time,mdark_in_2,'-',color= 'purple',zorder=10)
	ax.plot(time,mdark_in_5,'-',color= 'black' , label='DM',zorder=10)

	ax.plot(timebins[0:-1]+binwidth,mg01_mean,'--',color='red'   ,alpha=0.6,zorder=20)
	ax.plot(timebins[0:-1]+binwidth,mg02_mean,'--',color='orange',alpha=0.6,zorder=20)
	ax.plot(timebins[0:-1]+binwidth,mg05_mean,'--',color='green' ,alpha=0.6,zorder=20)
	ax.plot(timebins[0:-1]+binwidth,mg1_mean ,'--',color='blue'  ,alpha=0.6,zorder=20)
	ax.plot(timebins[0:-1]+binwidth,mg2_mean ,'--',color='purple',alpha=0.6,zorder=20)
	ax.plot(timebins[0:-1]+binwidth,mg5_mean ,'--',color='black' ,alpha=0.6,label='gas',zorder=20)

	s = np.where(models == sim)[0][0]
	
	leg = ax.legend(frameon=True,fancybox=False,loc='upper right',prop={'size':11})
	leg.set_zorder(100)

	text = ax.annotate(models_label[s],xy=(0.3,0.9),xycoords='axes fraction',fontsize=14,color='black',zorder=50)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

	text = ax.annotate('m < 0.1 kpc',xy=(0.02,1e6),xycoords='data',fontsize=11,color='red',zorder=50)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='red', 
		edgecolor='white'), path_effects.Normal()])

	text = ax.annotate('m < 0.2 kpc',xy=(0.02,8e6),xycoords='data',fontsize=11,color='orange',zorder=50)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='orange', 
		edgecolor='white'), path_effects.Normal()])

	text = ax.annotate('m < 0.5 kpc',xy=(0.02,7e7),xycoords='data',fontsize=11,color='green',zorder=50)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='green', 
		edgecolor='white'), path_effects.Normal()])

	text = ax.annotate('m < 1 kpc',xy=(0.02,3e8),xycoords='data',fontsize=11,color='blue',zorder=50)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='blue', 
		edgecolor='white'), path_effects.Normal()])

	text = ax.annotate('m < 2 kpc',xy=(0.02,1e9),xycoords='data',fontsize=11,color='purple',zorder=50)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='purple', 
		edgecolor='white'), path_effects.Normal()])

	text = ax.annotate('m < 5 kpc',xy=(0.02,3e9),xycoords='data',fontsize=11,color='black',zorder=50)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])


	ax.set_xlabel(r'time [$h^{-1}$ Gyr]')
	ax.set_ylabel(r'mass [M$_\odot$]')
	ax.set_yscale('log')
	ax.set_ylim(1e5,1e10)
	ax.set_xlim(0.05,1)

	# plt.savefig('/home/ejahn003/testfig_'+sim+'.png')

	p.finalize('mass_in_r_'+sim,save=1)

def plot_core_radius():
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	snapdir = '/mainvol/ejahn/smuggle/output/live_halo_02.2020/'
	h=0.7
	# drange = np.logspace(-1,2,100)
	

	colors = np.array(['black','purple','green','blue','orange','red'])

	print(d.datdir)
	corefile = h5py.File(d.datdir+'core_radius_new.hdf5','w')

	for sim in models:
		i = np.where(models == sim)[0][0]
		print(sim)

		rhofile = h5py.File(d.datdir+'rhodm_'+sim+'_new.hdf5','r')
		drange = np.array(rhofile['drange'])

		all_time = np.array([])
		all_core_radius = np.array([])

		for n in np.arange(1,201):
			rhodm = np.array(rhofile['snap_'+str(n).zfill(3)])

			#first snapshot is baseline so it cannot have a core radius
			if n==1:
				initial_rhodm = rhodm

			#calculate core radius for all snapshots >= 2	
			else:
				rho_ratio = initial_rhodm / rhodm
				sel_in_2 = (rho_ratio > 1.4) & (rho_ratio < 1.8)
				is_2 = np.count_nonzero(sel_in_2)

				if is_2==0:
					core_radius=0
				else:
					ind = np.where(sel_in_2)[0][-1]
					core_radius = drange[ind]

				all_core_radius = np.append(all_core_radius,core_radius)
				fname = snapdir + sim + '/snapshot_' + str(n).zfill(3)
				header = snapHDF5.snapshot_header(fname)
				all_time = np.append(all_time, header.time)

		timebins = np.linspace(0, 1, 70)
		binwidth = (timebins[1] - timebins[0])/2

		coreradius_mean = np.array([])

		for j in range(len(timebins)-1):
			leftbin = timebins[j]
			rightbin = timebins[j+1]

			sel = (all_time > leftbin) & (all_time < rightbin)
			coreradius_mean = np.append(coreradius_mean, np.mean(all_core_radius[sel]))

		ax.plot(timebins[0:-1]+binwidth,coreradius_mean,colors[i],label=models_label[i],lw=2,alpha=0.8)

		corefile.create_dataset(sim+'_coreradius',data=all_core_radius)
		corefile.create_dataset(sim+'_coreradius_smooth',data=coreradius_mean)
		corefile.create_dataset(sim+'_alltime',data=all_time)
		corefile.create_dataset(sim+'_time_smooth',data=timebins[0:-1]+binwidth)

			
	rhofile.close()
	corefile.close()
	#----------------------------------------------------------------------------------------------
	ax.legend(prop={'size':11},loc='upper left')#,frameon=False)

	ax.set_xlim(0.05,1)
	ax.set_xlabel(r'time [$h^{-1}$ Gyr]')
	ax.set_xticks([0.2,0.4,0.6,0.8,1])
	ax.set_xticks([0.1,0.3,0.5,0.7,0.9],minor=True)

	# ax.set_yscale('log')
	ax.set_ylim(0,0.8)
	ax.set_ylabel('core radius [kpc]')
	ax.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
	ax.set_yticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75],minor=True)

	p.finalize(fig,fname='core_radius_time_new',save=1)

def test_core_radius(n,sim):
	if n==1:
		raise ValueError('snapshot must be greater than 001')

	fig, ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	drange = np.logspace(-1,2,100)
	rhofile = h5py.File(d.datdir+'rhodm.hdf5','r')
	rhodm = np.array(rhofile[sim+'_rho'+str(n).zfill(3)])
	rhodm_0 = np.array(rhofile[sim+'_rho001'])

	corefile = h5py.File(d.datdir+'core_radius.hdf5','r')
	coreradius = np.array(corefile[sim+'_coreradius'])

	# all_time = np.array([])
	snaplist = np.arange(2,201)
	# for snap in snaplist:
	# 	fname = d.smuggledir + sim + '/snapshot_' + str(n).zfill(3)
	# 	header = snapHDF5.snapshot_header(fname)
	# 	all_time = np.append(all_time, header.time)

	this_coreradius = coreradius[np.where(snaplist == n)[0][0]]

	ax.plot(drange,rhodm_0,'k',lw=2,label='initial')
	ax.plot(drange,rhodm,'b',lw=2,label='snapshot '+str(n).zfill(3))
	ax.axvline(this_coreradius)
	ax.legend(prop={'size':11})

	ax.set_xscale('log')
	ax.set_xlim(1e-1,5e0)
	ax.set_xlabel('distance [kpc]')
	ax.set_xticks([0.1,0.2,0.5,1,2,5])
	ax.set_xticklabels([0.1,0.2,0.5,1,2,5])

	ax.set_yscale('log')
	ax.set_ylim(2e7,7e8)
	ax.set_ylabel(r'density [M$_\odot$ kpc$^\mathregular{-3}$]')

	rhofile.close()
	corefile.close()

	p.finalize(fname='rhocore_'+sim+'_'+str(n).zfill(3),save=1)

def plot_sigma_old(q):
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	
	colors = np.array(['black','purple','blue','green','orange','red'])
	distances = np.array([0.1,0.2,0.5,1,2,5])
	
	

	for sim in models:
		i = np.where(models==sim)[0][0]
		print(sim)
		sigmafile = h5py.File(d.datdir+'sigma_v_'+sim+'.hdf5','r')

		all_time = np.array([])
		for n in np.arange(10,201):
			fname = d.smuggledir + sim + '/snapshot_' + str(n).zfill(3)
			header = snapHDF5.snapshot_header(fname)
			all_time = np.append(all_time, header.time)

			sigma_star = np.array(sigmafile['sigma_star_'+str(n).zfill(3)])
			sigma_gas = np.array(sigmafile['sigma_gas_'+str(n).zfill(3)])

			if n==10:
				all_sigma_star = sigma_star
				all_sigma_gas = sigma_gas
			else:
				all_sigma_star = np.vstack((all_sigma_star,sigma_star))
				all_sigma_gas = np.vstack((all_sigma_gas,sigma_gas))

		# ax.plot(all_time,)
		# print(all_sigma_star.shape)
		# print(all_sigma_gas.shape)
		# for j in range(len(distances)):
		# 	# print(len(all_time),len(all_sigma_star[:,j]))
		# 	ax.plot(all_time,all_sigma_star[:,j],c=colors[i])

		ax.plot(all_time,all_sigma_star[:,q],c=colors[i],label=models_label[i])
		ax.plot(all_time,all_sigma_gas[:,q],'--',c=colors[i])
		sigmafile.close()
	
	ax.legend()
	ax.set_title('mass within '+str(distances[q])+' kpc   ',loc='right',size=10)
	ax.set_xlabel(r'time [$h^{-1}$ Gyr]')
	ax.set_ylabel(r'$\sigma_\ast$ [km s$^{-1}$]')
	ax.set_yscale('log')
	p.finalize(fname='sigma_'+str(distances[q]),save=1)

def plot_sigma(sim,dcut):
	fig,axarr = p.makefig('3_horiz',figx=15,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	
	colors = np.array(['black','purple','blue','green','orange','red'])
	distances = np.array([0.1,0.2,0.5,1,2,5])
	sigmafile = h5py.File(d.datdir+'sigma_v_'+sim+'.hdf5','r')
	timefile = h5py.File(d.datdir+'timefile.hdf5','r')

	if not(dcut in distances):
		raise ValueError('please choose a value from [0.1, 0.2, 0.5, 1, 2, 5]')
	else:
		d_ind = np.where(distances==dcut)[0][0]

	all_time = np.array(timefile['alltime_'+sim])[9:]

	for n in np.arange(10,201):
		# fname = d.smuggledir + sim + '/snapshot_' + str(n).zfill(3)
		# header = snapHDF5.snapshot_header(fname)
		# all_time = np.append(all_time, header.time)

		sigma_star = np.array(sigmafile['sigma_star_'+str(n).zfill(3)])[d_ind,:]
		sigma_gas = np.array(sigmafile['sigma_gas_'+str(n).zfill(3)])[d_ind,:]

		if n==10:
			all_sigma_star = sigma_star
			all_sigma_gas = sigma_gas
		else:
			all_sigma_star = np.vstack((all_sigma_star,sigma_star))
			all_sigma_gas = np.vstack((all_sigma_gas,sigma_gas))

	timefile.close()
	sigmafile.close()

	axarr[0].plot(all_time,all_sigma_star[:,0],c=c.orange)
	axarr[1].plot(all_time,all_sigma_star[:,1],c=c.orange)
	axarr[2].plot(all_time,all_sigma_star[:,2],c=c.orange)

	axarr[0].plot(all_time,all_sigma_gas[:,0],c=c.blue)
	axarr[1].plot(all_time,all_sigma_gas[:,1],c=c.blue)
	axarr[2].plot(all_time,all_sigma_gas[:,2],c=c.blue)

	axarr[1].set_xlabel(r'time [$\mathregular{h^{-1}}$ Gyr]')
	axarr[0].set_ylabel(r'$\sigma_i$ [km s$^{-1}$]')

	axarr[0].annotate(r'$\sigma_r$',xy=(0.85,0.9),xycoords='axes fraction',fontsize=25,color='black')
	axarr[1].annotate(r'$\sigma_\phi$',xy=(0.85,0.9),xycoords='axes fraction',fontsize=25,color='black')
	axarr[2].annotate(r'$\sigma_z$',xy=(0.85,0.9),xycoords='axes fraction',fontsize=25,color='black')

	label = models_label[np.where(models==sim)[0][0]]+'\n d < '+str(dcut)+'kpc'
	axarr[0].annotate(label,xy=(0.1,0.15),xycoords='axes fraction',fontsize=15,color='black')

	axarr[0].annotate('stars',xy=(0.1,0.9),xycoords='axes fraction',fontsize=15,color=c.orange)
	axarr[0].annotate('gas',xy=(0.1,0.84),xycoords='axes fraction',fontsize=15,color=c.blue)

	for i in np.arange(3):
		axarr[i].set_xticks([0.2,0.4,0.6,0.8,1])
		axarr[i].set_xticks([0.1,0.3,0.5,0.7,0.9],minor=True)
		axarr[i].set_yscale('log')
		axarr[i].set_ylim(1e0,1e2)

	p.finalize(fig,fname='sigma_'+sim+'_dcut'+str(dcut),save=True)#,tight=False)
	# plt.savefig(d.plotdir+month+'/'+'sigma_'+sim+'_dcut'+str(dcut)+'.png',dpi=200,format='png')

def plot_sigma_annuli(sim,d_ind,smooth=False):
	fig,axarr = p.makefig('3_horiz',figx=15,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	
	colors = np.array(['black','purple','blue','green','orange','red'])
	distances = np.array([0.1,0.3,0.5,1,2,4])
	sigmafile = h5py.File(d.datdir+'sigma_v_annuli'+sim+'.hdf5','r')
	timefile = h5py.File(d.datdir+'timefile.hdf5','r')
	all_time = np.array(timefile['alltime_'+sim])[9:]
	timefile.close()

	if d_ind > 4:
		raise ValueError('please choose a value from [0,1,2,3,4]')

	for n in np.arange(10,201):
		sigma_star = np.array(sigmafile['sigma_star_'+str(n).zfill(3)])[d_ind,:]
		sigma_gas = np.array(sigmafile['sigma_gas_'+str(n).zfill(3)])[d_ind,:]
		sigma_coldgas = np.array(sigmafile['sigma_coldgas_'+str(n).zfill(3)])[d_ind,:]

		if n==10:
			all_sigma_star = sigma_star
			all_sigma_gas = sigma_gas
			all_sigma_coldgas = sigma_coldgas
		else:
			all_sigma_star = np.vstack((all_sigma_star,sigma_star))
			all_sigma_gas = np.vstack((all_sigma_gas,sigma_gas))
			all_sigma_coldgas = np.vstack((all_sigma_coldgas,sigma_coldgas))
	sigmafile.close()

	if smooth:
		timebins = np.linspace(0, 1, 75)
		binwidth = (timebins[1] - timebins[0])/2

		for j in range(len(timebins)-1):
			leftbin = timebins[j]
			rightbin = timebins[j+1]

			sel = (all_time > leftbin) & (all_time < rightbin)
			
			if j==0:
				mean_rho = np.mean(all_sigma_star[sel][:,0])
				mean_phi = np.mean(all_sigma_star[sel][:,1])
				mean_z = np.mean(all_sigma_star[sel][:,2])
				sigma_star_mean = np.array([mean_rho,mean_phi,mean_z])

				mean_rho = np.mean(all_sigma_gas[sel][:,0])
				mean_phi = np.mean(all_sigma_gas[sel][:,1])
				mean_z = np.mean(all_sigma_gas[sel][:,2])
				sigma_gas_mean = np.array([mean_rho,mean_phi,mean_z])

				mean_rho_cold = np.mean(all_sigma_coldgas[sel][:,0])
				mean_phi_cold = np.mean(all_sigma_coldgas[sel][:,1])
				mean_z_cold = np.mean(all_sigma_coldgas[sel][:,2])
				sigma_coldgas_mean = np.array([mean_rho_cold,mean_phi_cold,mean_z_cold])
			else:
				mean_rho = np.mean(all_sigma_star[sel][:,0])
				mean_phi = np.mean(all_sigma_star[sel][:,1])
				mean_z = np.mean(all_sigma_star[sel][:,2])
				sigma_star_mean = np.vstack((sigma_star_mean,np.array([mean_rho,mean_phi,mean_z])))

				mean_rho = np.mean(all_sigma_gas[sel][:,0])
				mean_phi = np.mean(all_sigma_gas[sel][:,1])
				mean_z = np.mean(all_sigma_gas[sel][:,2])
				sigma_gas_mean = np.vstack((sigma_gas_mean,np.array([mean_rho,mean_phi,mean_z])))

				mean_rho_cold = np.mean(all_sigma_coldgas[sel][:,0])
				mean_phi_cold = np.mean(all_sigma_coldgas[sel][:,1])
				mean_z_cold = np.mean(all_sigma_coldgas[sel][:,2])
				sigma_coldgas_mean = np.vstack((sigma_coldgas_mean,np.array([mean_rho_cold,mean_phi_cold,mean_z_cold])))
		timeplt = timebins[0:-1]+binwidth
		sigmastarplt = sigma_star_mean
		sigmagasplt = sigma_gas_mean
		sigmacoldgasplt = sigma_coldgas_mean
	else:
		timeplt = all_time
		sigmastarplt = all_sigma_star
		sigmagasplt = all_sigma_gas
		sigmacoldgasplt = all_sigma_coldgas

	#---plot-the-things----------------------------------------------------
	for i in [0,1,2]:
		axarr[i].plot(timeplt,sigmastarplt[:,i],c=c.orange,alpha=0.7)
		axarr[i].plot(timeplt,sigmagasplt[:,i],c=c.blue,alpha=0.7)
		axarr[i].plot(timeplt,sigmacoldgasplt[:,i],c=c.green,alpha=0.7,ls=':')
		axarr[i].set_xticks([0.2,0.4,0.6,0.8,1])
		axarr[i].set_xticks([0.1,0.3,0.5,0.7,0.9],minor=True)
		# axarr[i].set_yscale('log')
		axarr[i].set_ylim(1e0,6e1)

	axarr[1].set_xlabel(r'time [$\mathregular{h^{-1}}$ Gyr]')
	# axarr[1].set_xlabel(r'time [$h^{-1}$ Gyr]')
	axarr[0].set_ylabel(r'$\mathregular{\sigma_i}$ [km s$^{-1}$]')

	text = axarr[0].annotate(r'$\mathregular{\sigma_r}$',xy=(0.85,0.9),xycoords='axes fraction',fontsize=25,color='black')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])
	text = axarr[1].annotate(r'$\mathregular{\sigma_\phi}$',xy=(0.85,0.9),xycoords='axes fraction',fontsize=25,color='black')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])
	text = axarr[2].annotate(r'$\mathregular{\sigma_z}$',xy=(0.85,0.9),xycoords='axes fraction',fontsize=25,color='black')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

	# if d_ind==4:
	label = models_label[np.where(models==sim)[0][0]]+'\n'+str(distances[d_ind])+' < r < '+str(distances[d_ind+1])
	text = axarr[1].annotate(label,xy=(0.2,0.075),xycoords='axes fraction',fontsize=15,color='black')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

	text = axarr[0].annotate('stars',xy=(0.07,0.92),xycoords='axes fraction',fontsize=15,color=c.orange)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.orange, 
		edgecolor='white'), path_effects.Normal()])
	text = axarr[0].annotate('gas',xy=(0.07,0.87),xycoords='axes fraction',fontsize=15,color=c.blue)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.blue, 
		edgecolor='white'), path_effects.Normal()])
	text = axarr[0].annotate('cold gas',xy=(0.07,0.82),xycoords='axes fraction',fontsize=15,color=c.green)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.green, 
		edgecolor='white'), path_effects.Normal()])

	# for i in np.arange(3):
		

	p.finalize(fig,fname='sigma/sigma_annuli_'+sim+'_'+str(d_ind),save=1)#,tight=False)
	# plt.savefig(d.plotdir+month+'/'+'sigma_'+sim+'_dcut'+str(dcut)+'.png',dpi=200,format='png')

def plot_sigma_profile_all(sim,snapnum=400):
	# fig,ax = p.makefig(1)
	fig,axarr = p.makefig('3_horiz',figx=15,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	if not(sim in models):
		raise ValueError('please choose a sim in models')

	j = np.where(models==sim)[0][0]

	sigmaname = d.datdir+'sigma_profile_'+savenames[j]+'.hdf5'
	sigmafile = h5py.File(sigmaname,'r')
	sigma_star = np.array(sigmafile['sigma_star'])
	sigma_gas = np.array(sigmafile['sigma_gas'])
	sigma_coldgas = np.array(sigmafile['sigma_coldgas'])
	sigmafile.close()

	dbins = np.append(0.,np.logspace(-1,0.7,30))
	dplt = np.array([])

	for i in np.arange(len(dbins)-1):
		# left = dbins[i]
		# right = dbins[i+1]
		mean = np.mean(np.array([dbins[i],dbins[i+1]]))
		dplt = np.append(dplt,mean)

	sigma_star_rho = sigma_star[:,0]
	sigma_star_phi = sigma_star[:,1]
	sigma_star_z   = sigma_star[:,2]

	sigma_gas_rho = sigma_gas[:,0]
	sigma_gas_phi = sigma_gas[:,1]
	sigma_gas_z   = sigma_gas[:,2]

	sigma_coldgas_rho = sigma_coldgas[:,0]
	sigma_coldgas_phi = sigma_coldgas[:,1]
	sigma_coldgas_z   = sigma_coldgas[:,2]

	axarr[0].plot(dplt,sigma_star_rho,ls='-',c=c.orange,alpha=0.8,lw=2)
	axarr[1].plot(dplt,sigma_star_phi,ls='-',c=c.orange,alpha=0.8,lw=2)
	axarr[2].plot(dplt,sigma_star_z,ls='-',c=c.orange,alpha=0.8,lw=2)
	# axarr[0].plot(1e3,1e3,ls='-',c='black',label='stars')

	axarr[0].plot(dplt,sigma_gas_rho,ls='-',c=c.blue,alpha=0.8,lw=2)
	axarr[1].plot(dplt,sigma_gas_phi,ls='-',c=c.blue,alpha=0.8,lw=2)
	axarr[2].plot(dplt,sigma_gas_z,ls='-',c=c.blue,alpha=0.8,lw=2)
	# axarr[0].plot(1e3,1e3,ls='--',c='black',label='gas')

	axarr[0].plot(dplt,sigma_coldgas_rho,ls='--',c=c.green,alpha=0.8,lw=1.5)
	axarr[1].plot(dplt,sigma_coldgas_phi,ls='--',c=c.green,alpha=0.8,lw=1.5)
	axarr[2].plot(dplt,sigma_coldgas_z,ls='--',c=c.green,alpha=0.8,lw=1.5)
	# axarr[0].plot(1e3,1e3,ls=':',lw=1.5,c='black',label='cold gas')

	text = axarr[0].annotate(r'$\mathregular{\sigma_r}$',xy=(0.9,0.9),xycoords='axes fraction',fontsize=20,color='k')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

	text = axarr[1].annotate(r'$\mathregular{\sigma_\phi}$',xy=(0.9,0.9),xycoords='axes fraction',fontsize=20,color='k')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

	text = axarr[2].annotate(r'$\mathregular{\sigma_z}$',xy=(0.9,0.9),xycoords='axes fraction',fontsize=20,color='k')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])
	
	simtext = models_label[j]+'\nsnapshot '+str(snapnum).zfill(3)
	text = axarr[1].annotate(simtext,xy=(0.25,0.85),xycoords='axes fraction',fontsize=12,color='black')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

	text = axarr[0].annotate('stars',xy=(0.07,0.92),xycoords='axes fraction',fontsize=15,color=c.orange)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.orange, 
		edgecolor='white'), path_effects.Normal()])
	text = axarr[0].annotate('gas',xy=(0.07,0.87),xycoords='axes fraction',fontsize=15,color=c.blue)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.blue, 
		edgecolor='white'), path_effects.Normal()])
	text = axarr[0].annotate('cold gas',xy=(0.07,0.82),xycoords='axes fraction',fontsize=15,color=c.green)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.green, 
		edgecolor='white'), path_effects.Normal()])
	
	# ax.legend(loc='upper center',prop={'size':13},frameon=False)

	axarr[0].set_ylabel(r'$\mathregular{\sigma_i}$ [km s$\mathregular{^{-1}}$]')
	axarr[1].set_xlabel('d [kpc]')

	for i in [0,1,2]:
		axarr[i].set_xscale('log')
		p.clear_axes(axarr[i])

		axarr[i].set_xlim(0.1,5)
		axarr[i].set_xticks([0.1,0.2,0.5,1.,2.])
		axarr[i].set_xticklabels([0.1,0.2,0.5,1.,2.])
		# ax.set_xticks([0,1,2,3,4,5])
		# ax.set_xticks([0.5,1.5,2.5,3.5,4.5],minor=True)
		
		axarr[i].set_ylim(0,62)
		axarr[i].set_yticks([10,20,30,40,50,60])
		axarr[i].set_yticks([5,15,25,35,45,55],minor=True)

	p.finalize(fig,fname='sigma_profile_'+savenames[j],save=1)

def plot_sigma_radial_single(sim,snapnum=400):
	fig,ax = p.makefig(1,figx=8,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	if not(sim in models):
		raise ValueError('please choose a sim in models')

	j = np.where(models==sim)[0][0]


	#--read-sigma-for-current-model----------------------------------------------
	sigmaname = d.datdir+'sigma_profile_'+savenames[j]+'.hdf5'
	sigmafile = h5py.File(sigmaname,'r')
	sigma_star = np.array(sigmafile['sigma_star'])
	sigma_gas = np.array(sigmafile['sigma_gas'])
	sigma_coldgas = np.array(sigmafile['sigma_coldgas'])
	sigmafile.close()

	dbins = np.append(0.,np.logspace(-1,0.7,30))
	dplt = np.array([])

	for i in np.arange(len(dbins)-1):
		# left = dbins[i]
		# right = dbins[i+1]
		mean = np.mean(np.array([dbins[i],dbins[i+1]]))
		dplt = np.append(dplt,mean)

	sigma_star_rho = sigma_star[:,0]
	sigma_gas_rho = sigma_gas[:,0]
	sigma_coldgas_rho = sigma_coldgas[:,0]

	ax.plot(dplt,sigma_star_rho,ls='-',c=c.orange,alpha=0.8,lw=2,zorder=13)
	ax.plot(dplt,sigma_gas_rho,ls=':',c=c.blue,alpha=0.8,lw=2,zorder=20)
	ax.plot(dplt,sigma_coldgas_rho,ls='--',c=c.green,alpha=0.8,lw=1.5,zorder=10)

	#--read-sigma-for-fiducial-----------------------------------------------------
	sigmaname = d.datdir+'sigma_profile_fiducial_1e5.hdf5'
	sigmafile = h5py.File(sigmaname,'r')
	sigma_star = np.array(sigmafile['sigma_star'])
	sigma_gas = np.array(sigmafile['sigma_gas'])
	sigma_coldgas = np.array(sigmafile['sigma_coldgas'])
	sigmafile.close()

	dbins = np.append(0.,np.logspace(-1,0.7,30))
	dplt = np.array([])

	for i in np.arange(len(dbins)-1):
		# left = dbins[i]
		# right = dbins[i+1]
		mean = np.mean(np.array([dbins[i],dbins[i+1]]))
		dplt = np.append(dplt,mean)

	sigma_star_rho = sigma_star[:,0]
	sigma_gas_rho = sigma_gas[:,0]
	sigma_coldgas_rho = sigma_coldgas[:,0]

	ax.plot(dplt,sigma_star_rho,ls='-',c='k',alpha=0.2,lw=1,zorder=1)
	ax.plot(dplt,sigma_gas_rho,ls=':',c='k',alpha=0.2,lw=1,zorder=1)
	ax.plot(dplt,sigma_coldgas_rho,ls='--',c='k',alpha=0.2,lw=1,zorder=1)

	#--legends-and-stuff-----------------------------------------------------------
	# text = ax.annotate(r'$\mathregular{\sigma_r}$',xy=(0.9,0.9),xycoords='axes fraction',fontsize=20,color='k')
	# text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
	# 	edgecolor='white'), path_effects.Normal()])

	simtext = models_label[j]+'\nsnapshot '+str(snapnum).zfill(3)
	text = ax.annotate(simtext,xy=(0.6,0.85),xycoords='axes fraction',fontsize=12,color='black')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

	text = ax.annotate('stars',xy=(0.07,0.92),xycoords='axes fraction',fontsize=15,color=c.orange)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.orange, 
		edgecolor='white'), path_effects.Normal()])
	text = ax.annotate('gas',xy=(0.07,0.87),xycoords='axes fraction',fontsize=15,color=c.blue)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.blue, 
		edgecolor='white'), path_effects.Normal()])
	text = ax.annotate('cold gas',xy=(0.07,0.82),xycoords='axes fraction',fontsize=15,color=c.green)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.green, 
		edgecolor='white'), path_effects.Normal()])
	text = ax.annotate('fiducial',xy=(0.07,0.77),xycoords='axes fraction',fontsize=15,color='grey')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='grey', 
		edgecolor='white'), path_effects.Normal()])
	# ax.legend(loc='upper center',prop={'size':13},frameon=False)

	ax.set_ylabel(r'$\mathregular{\sigma_r}$ [km s$\mathregular{^{-1}}$]')
	ax.set_xlabel('d [kpc]')

	ax.set_xscale('log')
	p.clear_axes(ax)

	ax.set_xlim(0.1,5)
	ax.set_xticks([0.1,0.2,0.5,1.,2.])
	ax.set_xticklabels([0.1,0.2,0.5,1.,2.])
	# ax.set_xticks([0,1,2,3,4,5])
	# ax.set_xticks([0.5,1.5,2.5,3.5,4.5],minor=True)
	
	ax.set_ylim(0,50)
	ax.set_yticks([10,20,30,40,50])
	ax.set_yticks([5,15,25,35,45],minor=True)

	p.finalize(fig,fname='sigma_radial_profile_'+savenames[j],save=1)

def plot_sigma_radial_all(snapnum):
	fig,ax = p.makefig(1,figx=8,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for sim in models:
		i = np.where(models==sim)[0][0]


		#--read-sigma-for-current-model----------------------------------------------
		sigmaname = d.datdir+'sigma_profiles_'+savenames[i]+'.hdf5'
		sigmafile = h5py.File(sigmaname,'r')
		sigma_star = np.array(sigmafile['sigma_star_'+str(snapnum).zfill(3)])
		sigma_gas = np.array(sigmafile['sigma_gas_'+str(snapnum).zfill(3)])
		sigma_coldgas = np.array(sigmafile['sigma_coldgas_'+str(snapnum).zfill(3)])
		sigmafile.close()

		dbins = np.append(0.,np.logspace(-1,0.7,30))
		dplt = np.array([])

		for j in np.arange(len(dbins)-1):
			# left = dbins[i]
			# right = dbins[i+1]
			mean = np.mean(np.array([dbins[j],dbins[j+1]]))
			dplt = np.append(dplt,mean)

		sigma_star_rho = sigma_star[:,0]
		# sigma_gas_rho = sigma_gas[:,0]
		# sigma_coldgas_rho = sigma_coldgas[:,0]

		ax.plot(dplt,sigma_star_rho,ls='-',c=colors_list[i],alpha=0.8,lw=2)#,zorder=13)

		simtext = models_label[i]#+'\nsnapshot '+str(snapnum).zfill(3)
		text = ax.annotate(simtext,xy=(0.1,0.95-(i*0.05)),xycoords='axes fraction',fontsize=12,color=colors_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=colors_list[i], 
			edgecolor='white'), path_effects.Normal()])

	ax.set_ylabel(r'$\mathregular{\sigma_r}$(stars) [km s$\mathregular{^{-1}}$]')
	ax.set_xlabel('d [kpc]')

	ax.set_xscale('log')
	p.clear_axes(ax)

	ax.set_xlim(0.1,5)
	ax.set_xticks([0.1,0.2,0.5,1.,2.,5.])
	ax.set_xticklabels([0.1,0.2,0.5,1.,2.,5.])

	ax.set_ylim(0,50)
	ax.set_yticks([10,20,30,40,50])
	ax.set_yticks([5,15,25,35,45],minor=True)

	p.finalize(fig,'sigma_r_star_'+whichsims,save=1)
	
def plot_sigma_disk(sim,snapnum=400):
	fig,ax = p.makefig(1,figx=8,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	if not(sim in models):
		raise ValueError('please choose a sim in models')

	j = np.where(models==sim)[0][0]


	#--read-sigma-for-current-model----------------------------------------------
	sigmaname = d.datdir+'sigma_profile_'+savenames[j]+'.hdf5'
	sigmafile = h5py.File(sigmaname,'r')
	sigma_star = np.array(sigmafile['sigma_star'])
	sigma_gas = np.array(sigmafile['sigma_gas'])
	sigma_coldgas = np.array(sigmafile['sigma_coldgas'])
	sigmafile.close()

	dbins = np.append(0.,np.logspace(-1,0.7,30))
	dplt = np.array([])

	for i in np.arange(len(dbins)-1):
		# left = dbins[i]
		# right = dbins[i+1]
		mean = np.mean(np.array([dbins[i],dbins[i+1]]))
		dplt = np.append(dplt,mean)

	sigma_star_phiz = sigma_star[:,1]/sigma_star[:,2]
	sigma_gas_phiz = sigma_gas[:,1]/sigma_gas[:,2]
	sigma_coldgas_phiz = sigma_coldgas[:,1]/sigma_coldgas[:,2]


	ax.plot(dplt,sigma_star_phiz,ls='-',c=c.orange,alpha=0.8,lw=2,zorder=13)
	ax.plot(dplt,sigma_gas_phiz,ls=':',c=c.blue,alpha=0.8,lw=2,zorder=20)
	ax.plot(dplt,sigma_coldgas_phiz,ls='--',c=c.green,alpha=0.8,lw=1.5,zorder=10)

	#--read-sigma-for-fiducial-----------------------------------------------------
	sigmaname = d.datdir+'sigma_profile_fiducial_1e5.hdf5'
	sigmafile = h5py.File(sigmaname,'r')
	sigma_star = np.array(sigmafile['sigma_star'])
	sigma_gas = np.array(sigmafile['sigma_gas'])
	sigma_coldgas = np.array(sigmafile['sigma_coldgas'])
	sigmafile.close()

	dbins = np.append(0.,np.logspace(-1,0.7,30))
	dplt = np.array([])

	for i in np.arange(len(dbins)-1):
		# left = dbins[i]
		# right = dbins[i+1]
		mean = np.mean(np.array([dbins[i],dbins[i+1]]))
		dplt = np.append(dplt,mean)

	sigma_star_phiz = sigma_star[:,1]/sigma_star[:,2]
	sigma_gas_phiz = sigma_gas[:,1]/sigma_gas[:,2]
	sigma_coldgas_phiz = sigma_coldgas[:,1]/sigma_coldgas[:,2]

	ax.plot(dplt,sigma_star_phiz,ls='-',c='k',alpha=0.2,lw=1,zorder=1)
	ax.plot(dplt,sigma_gas_phiz,ls=':',c='k',alpha=0.2,lw=1,zorder=1)
	ax.plot(dplt,sigma_coldgas_phiz,ls='--',c='k',alpha=0.2,lw=1,zorder=1)

	#--legends-and-stuff-----------------------------------------------------------
	# text = ax.annotate(r'$\mathregular{\sigma_r}$',xy=(0.9,0.9),xycoords='axes fraction',fontsize=20,color='k')
	# text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
	# 	edgecolor='white'), path_effects.Normal()])

	simtext = models_label[j]+'\nsnapshot '+str(snapnum).zfill(3)
	text = ax.annotate(simtext,xy=(0.3,0.85),xycoords='axes fraction',fontsize=12,color='black')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

	text = ax.annotate('stars',xy=(0.07,0.92),xycoords='axes fraction',fontsize=15,color=c.orange)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.orange, 
		edgecolor='white'), path_effects.Normal()])
	text = ax.annotate('gas',xy=(0.07,0.87),xycoords='axes fraction',fontsize=15,color=c.blue)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.blue, 
		edgecolor='white'), path_effects.Normal()])
	text = ax.annotate('cold gas',xy=(0.07,0.82),xycoords='axes fraction',fontsize=15,color=c.green)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=c.green, 
		edgecolor='white'), path_effects.Normal()])
	text = ax.annotate('fiducial',xy=(0.07,0.77),xycoords='axes fraction',fontsize=15,color='grey')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='grey', 
		edgecolor='white'), path_effects.Normal()])
	# ax.legend(loc='upper center',prop={'size':13},frameon=False)

	ax.set_ylabel(r'$\mathregular{\sigma_{\phi}/\sigma_{z}}$')
	ax.set_xlabel('d [kpc]')

	ax.set_xscale('log')
	p.clear_axes(ax)

	ax.set_xlim(0.1,5)
	ax.set_xticks([0.1,0.2,0.5,1.,2.,5.])
	ax.set_xticklabels([0.1,0.2,0.5,1.,2.,5.])
	# ax.set_xticks([0,1,2,3,4,5])
	# ax.set_xticks([0.5,1.5,2.5,3.5,4.5],minor=True)
	
	ax.set_ylim(0,6)
	# ax.set_yticks([10,20,30,40,50])
	# ax.set_yticks([5,15,25,35,45],minor=True)

	p.finalize(fig,fname='sigma_disk_profile_'+savenames[j],save=1)

def core_radius_sfr(sim):
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'
	h=0.7

	corefile = h5py.File(d.datdir+'core_radius.hdf5','r')
	rcore = np.array(corefile[sim+'_coreradius'])
	time_rc = np.array(corefile[sim+'_alltime'])
	corefile.close()

	filein = d.smuggledir + sim + '/sfr.txt'
	data = np.loadtxt(filein)
	time = data[:,0]
	sfr = data[:,2]

	kk = sfr == 0
	sfr[kk]=1.e-10
	ax.plot(time,sfr,color='b',linewidth=1.3,label='SFR')

	ax.set_yscale('log')
	ax.set_ylim([1.e-4,30])
	ax.set_xlim([0,1])
	ax.set_xlabel(r'time [$h^{-1}$Gyr]')
	ax.set_ylabel(r'SFR [M$_\odot$/yr]')

	ax2 = ax.twinx()

	ax2.plot(time_rc,rcore,'-',c='orange',lw=1.3,label=r'$r_\mathregular{core}$')
	ax2.set_ylim(0,0.7)
	ax2.set_xlim([0,1])
	ax2.set_ylabel(r'$r_\mathregular{core}$ [kpc]')
	ax2.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
	ax2.set_yticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65],minor=True)

	ax.annotate('SFR',xy=(0.05,0.95),xycoords='axes fraction',fontsize=11,color='b')
	ax.annotate(r'$r_\mathregular{core}$',xy=(0.05,0.90),xycoords='axes fraction',fontsize=16,color='orange')
	ax.annotate(models_label[np.where(models==sim)[0][0]],xy=(0.65,0.90),xycoords='axes fraction',fontsize=14,color='black')


	p.finalize(fig,fname='rcore_sfr_'+sim,save=1)

def gas_density(sim,dcut=0.5):
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	# snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'
	h=0.7

	if not(sim in models):
		raise ValueError('please choose an asbrbrbrb simuawuiwda')

	i = np.where(models==sim)[0][0]

	fname = d.datdir+'rhogas_fromsnap_'+savenames[i]+'.hdf5'
	gasfile = h5py.File(fname,'r')
	distances = np.array([0.1,0.2,0.5,1,2,5])
	colors = np.array(['red','green','blue','purple','black'])

	if not(dcut in distances):
		raise ValueError('please choose a dcut from this list: [0.1,0.2,0.5,1,2,5]')

	d_ind = np.where(distances==dcut)[0][0]

	f = h5py.File(d.datdir+'massprofiles_'+savenames[i]+'.hdf5','r')
	drange = np.array(f['drange'])

	core_radius = np.array([])

	#---read-and-plot-gas-density------------------------------------------------------------------
	all_time = np.array([])
	for snapnum in range(1,401):

		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
		header = snapHDF5.snapshot_header(snapfile)
		all_time = np.append(all_time, header.time)


		rhogas = np.array(gasfile['snap_'+str(snapnum).zfill(3)])
		rhogas_cold = np.array(gasfile['cold_snap_'+str(snapnum).zfill(3)])

		if snapnum==1:
			rhogas_all = rhogas
			rhogas_cold_all = rhogas_cold
		else:
			rhogas_all = np.vstack((rhogas_all,rhogas))
			rhogas_cold_all = np.vstack((rhogas_cold_all,rhogas_cold))

		#--calculate-core-radius-------------------------------------------------------------------
		dark_profile = np.array(f['dark'])[snapnum]
		dark_profile_0 = np.array(f['dark'])[0]
		vols = 4./3.*np.pi*(drange**3)
		density = dark_profile/vols
		density_0 = dark_profile_0/vols
		
		rho_ratio = density_0 / density
		sel_in_2 = (rho_ratio > 1.4) & (rho_ratio < 1.8)
		is_2 = np.count_nonzero(sel_in_2)

		if is_2==0:
			core_radius = np.append(core_radius,0.)
		else:
			ind = np.where(sel_in_2)[0][-1]
			core_radius = np.append(core_radius,drange[ind])


	#--plotting------------------------------------------------------------------------------------
	# for j in range(len(distances)-1):
	# 	this_rhogas = rhogas_all[:,j]

	# 	sel = (this_rhogas < 1e4)
	# 	this_rhogas[sel] = np.nan

	# 	ax.plot(all_time,this_rhogas,'-',lw=1.7,c=colors[j],zorder=500)
	# 	ax.annotate('d < '+str(distances[j])+' kpc',xy=(0.8,(0.95-j*0.05)),xycoords='axes fraction',fontsize=11,color=colors[j])

	ax.plot(all_time,rhogas_all[:,d_ind],'-',lw=1.7,c='blue',zorder=500)
	ax.annotate('all gas < '+str(dcut)+' kpc',xy=(0.7,0.95),xycoords='axes fraction',fontsize=11,color='blue')

	ax.plot(all_time,rhogas_cold_all[:,d_ind],'-',lw=1.7,c='green',zorder=500)
	ax.annotate('cold gas < '+str(dcut)+' kpc',xy=(0.7,0.90),xycoords='axes fraction',fontsize=11,color='green')

	ax.annotate('core radius',xy=(0.7,0.85),xycoords='axes fraction',fontsize=11,color='orange')



	#---plot-core-radius---------------------------------------------------------------------------
	ax2 = ax.twinx()
	# corefile = h5py.File(d.datdir+'core_radius.hdf5','r')
	# rcore = np.array(corefile[sim+'_coreradius'])
	# time_rc = np.array(corefile[sim+'_alltime'])
	# corefile.close()

	y0 = np.zeros(len(all_time))
	ax2.fill_between(all_time[5:],y0[5:],core_radius[5:],facecolor='orange', alpha=0.2, zorder=0, edgecolor='orange',hatch='/')

	ax2.plot(all_time[5:],core_radius[5:],'-',c='orange',lw=2.5,zorder=1)

	ax2.set_ylim(0,1.4)
	ax2.set_xlim([0,2])
	ax2.set_ylabel(r'$r_\mathregular{core}$ [kpc]')#,color='orange')
	# ax2.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
	# ax2.set_yticks([0.05,0.15,0.25,0.35,0.45,0.55,0.65],minor=True)


	#---finish-plotting----------------------------------------------------------------------------
	# ax.legend(loc='upper right')
	ax.annotate(models_label[i],xy=(0.05,0.95),xycoords='axes fraction',fontsize=11,color='black')

	ax.set_xlabel(r'time [$h^{-1}$ Gyr]')
	ax.set_xlim(0,2)
	ax.set_yscale('log')
	# ax.set_ylabel(r'$\rho_\mathregular{gas}$ [M$_\odot$ kpc$^{-3}$]')
	ax.set_ylabel(r'$\rho_\mathregular{gas}$ [cm$^{-3}$ $h^3$]')
	# ax.set_ylim(1e4,1e10)
	ax.set_ylim(1e-2,1e5)

	p.finalize(fig,fname='rhogas_fromsnap_'+str(dcut)+'kpc_'+savenames[i],save=1)

def rhogas_rcore_points(sim,nbins):
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	snapdir = '/mainvol/ejahn/smuggle/output/live_halo/'
	h=0.7

	fname = d.datdir+'rhogas_'+sim+'.hdf5'
	gasfile = h5py.File(fname,'r')
	simname = models_label[np.where(models==sim)[0][0]]
	distances = np.array([0.1,0.2,0.5,1,2,5])
	colors = np.array(['red','green','blue','purple','black'])

	#---read-gas-density---------------------------------------------------------------------------
	time_gas = np.array([])
	for snapnum in range(1,201):

		snapfile = snapdir+sim+'/snapshot_'+str(snapnum).zfill(3)
		header = snapHDF5.snapshot_header(snapfile)
		time_gas = np.append(time_gas, header.time)


		rhogas = np.array(gasfile['snap_'+str(snapnum).zfill(3)])
		rhogas_cold = np.array(gasfile['cold_snap_'+str(snapnum).zfill(3)])

		if snapnum==1:
			rhogas_all = rhogas
			rhogas_cold_all = rhogas_cold
		else:
			rhogas_all = np.vstack((rhogas_all,rhogas))
			rhogas_cold_all = np.vstack((rhogas_cold_all,rhogas_cold))


	rhogas_1kpc = rhogas_all[:,3]
	rhogas_cold_1kpc = rhogas_cold_all[:,3]

	#---read-core-radius---------------------------------------------------------------------------
	corefile = h5py.File(d.datdir+'core_radius.hdf5','r')
	rcore = np.array(corefile[sim+'_coreradius'])
	time_core = np.array(corefile[sim+'_alltime'])
	corefile.close()

	#----------------------------------------------------------------------------------------------
	timebins = np.linspace(0, 1, nbins)
	binwidth = (timebins[1] - timebins[0])/2
	
	rcore_b = np.array([])
	rhogas_b = np.array([])
	rhogas_cold_b = np.array([])

	for j in range(len(timebins)-1):
		leftbin = timebins[j]
		rightbin = timebins[j+1]

		sel = (time_core > leftbin) & (time_core < rightbin)
		rcore_b = np.append(rcore_b,np.mean(rcore[sel]))

		sel = (time_gas > leftbin) & (time_gas < rightbin)
		rhogas_b = np.append(rhogas_b,np.mean(rhogas_1kpc[sel]))
		rhogas_cold_b = np.append(rhogas_cold_b,np.mean(rhogas_cold_1kpc[sel]))

	ax.plot(rhogas_b,rcore_b,'s',label='all gas < 1 kpc')
	ax.plot(rhogas_cold_b,rcore_b,'o',label='cold gas < 1 kpc')	
	ax.legend(prop={'size':10})	
	ax.set_ylim(0,0.7)
	ax.set_xscale('log')
	ax.set_xlabel(r'$\rho_\mathregular{gas}$ [M$_\odot$ kpc$^{-3}$]')
	ax.set_ylabel(r'$r_\mathregular{core}$ [kpc]')

	p.finalize(fig,fname='rhogas_points_'+sim,save=1)

def scale_test():
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	numproc = np.array([2,4,8,16,32,64])
	snapdir = '/mainvol/ejahn/smuggle/output/scale_test/'
	year = 2020
	months = np.array(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

	from datetime import datetime

	comp_time = np.array([])

	for n in numproc:
		thisfile = snapdir + 'fiducial_1e5_np_' + str(n).zfill(2) + '/lsl.txt'
		dates = np.loadtxt(thisfile,dtype=str,skiprows=1,usecols=(5,6,7))
		fnames = np.loadtxt(thisfile,dtype=str,skiprows=1,usecols=(8,))

		i_ind = np.where(fnames=='snapshot_000.hdf5')[0][0]
		f_ind = np.where(fnames=='snapshot_040.hdf5')[0][0]

		date_i = dates[i_ind]
		date_f = dates[f_ind]

		month_i = np.where(months==date_i[0])[0][0]+1
		month_f = np.where(months==date_f[0])[0][0]+1

		#          datetime(year,  month,    day,       hour,     minute, second, microsecond)
		datetime_i = datetime(2020,int(month_i),int(date_i[1]),int(date_i[2][:2]),int(date_i[2][-2:]))
		datetime_f = datetime(2020,int(month_f),int(date_f[1]),int(date_f[2][:2]),int(date_f[2][-2:]))

		delta = datetime_f - datetime_i

		comp_time = np.append(comp_time,delta.total_seconds())

	ax.plot(numproc,comp_time,'-o',c='blue')
	
	ax.annotate('fiducial_1e5, static halo',xy=(0.6,0.9),xycoords='axes fraction',fontsize=11,color='blue')
	ax.annotate('run time = 200 Myr',xy=(0.6,0.85),xycoords='axes fraction',fontsize=11,color='blue')
	
	ax.set_yscale('log')
	ax.set_ylabel(r'$t_\mathregular{comp}$ (s)')

	ax.set_xscale('log')
	ax.set_xlabel(r'$n_\mathregular{proc}$')
	ax.set_xlim(1.5,100)
	ax.set_xticks([2,4,8,16,32,64])
	ax.set_xticklabels([2,4,8,16,32,64])


	p.finalize(fig,save=1)

def rotationcurve(snapnum):
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	h=0.7
	drange = np.logspace(-1,2,200)
	snapdirs = np.array(['/mainvol/ejahn/smuggle/output/live_halo_09.2019/fiducial_1e5/',
				'/mainvol/ejahn/smuggle/output/live_halo_02.2020/compact_dwarf/fiducial_1e5/',
				'/mainvol/ejahn/smuggle/output/live_halo_02.2020/vareff_1e5/'])

	colors = np.array(['blue','red','green'])
	labels = np.array(['fiducial 1e5','compact dwarf 1e5','var. eff. 1e5'])

	fname = 'rotationcurve_'+str(snapnum).zfill(3)

	nsim = len(snapdirs)

	for snapdir in snapdirs:
		i = np.where(snapdirs==snapdir)[0][0]
		snapfile = snapdir+'/snapshot_'+str(snapnum).zfill(3)
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
		d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)

		type2mass = snapHDF5.read_block(snapfile, 'MASS', parttype=2)*(1.e10)/h
		type2_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=2)/h
		d_type2 = np.linalg.norm(type2_pos-dark_cm, axis=1)

		type3mass = snapHDF5.read_block(snapfile, 'MASS', parttype=3)*(1.e10)/h
		type3_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=3)/h
		d_type3 = np.linalg.norm(type3_pos-dark_cm, axis=1)

		vcirc_profile = np.array([])

		for dist in drange:
			mass_in_d = np.sum(darkmass[(d_dark < dist)])
			mass_in_d += np.sum(gasmass[(d_gas < dist)])
			mass_in_d += np.sum(type2mass[(d_type2 < dist)])
			mass_in_d += np.sum(type3mass[(d_type3 < dist)])

			vcirc_in_d = np.sqrt(m.Gprime*mass_in_d/dist)
			vcirc_profile = np.append(vcirc_profile,vcirc_in_d)

		ax.plot(drange,vcirc_profile,c=colors[i])

	ax.set_xscale('log')
	ax.set_xlabel('r (kpc)')
	ax.set_xlim(1e-1,5e1)

	ax.set_yscale('log')
	ax.set_ylabel(r'$v_\mathregular{circ}$ [km s$^{-1}$]')

	for i in range(nsim):
		ax.annotate(labels[i],xy=(0.05,1-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors[i])

	ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.75,0.05),xycoords='axes fraction',fontsize=11,color='k')

			
	p.finalize(fig,fname,save=1)

def massprofile(sim,snapnum,do_density=False,do_velocity=False,plot_initial=False,print_radii=False,dicintio=False,do_slope=False,do_1e6=False):

	if do_density and do_velocity:
		raise ValueError('cannot plot both density and velocity, please choose one')

	if not(sim in models):
		raise ValueError('please use a sim in the list of models')

	i = np.where(models==sim)[0][0]

	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	ax.plot(1e20,1e20,'k',ls='-',label='snapshot '+str(snapnum).zfill(3))
	all_types = np.array(['total','dark matter','gas','type 2','type 3','type 4'])
	all_colors = np.array(['grey','black','blue','green','orange','red'])

	mask = np.array([True,True,True,True,True,True])
	all_types = all_types[mask]
	all_colors = all_colors[mask]
	thismassfile = savenames[i]

	f = h5py.File(d.datdir+'massprofiles_'+thismassfile+'.hdf5','r')
	print(d.datdir+'massprofiles_'+thismassfile+'.hdf5')
	drange = np.array(f['drange'])
	gas_profile = np.array(f['gas'])[snapnum]
	dark_profile = np.array(f['dark'])[snapnum]
	type2_profile = np.array(f['type2'])[snapnum]
	type3_profile = np.array(f['type3'])[snapnum]
	type4_profile = np.array(f['type4'])[snapnum]

	if plot_initial:
		dark_profile_0 = np.array(f['dark'])[0]
	f.close()

	vols = 4./3.*np.pi*(drange**3)
	total_profile = gas_profile + dark_profile + type2_profile + type3_profile + type4_profile

	all_profiles = np.array([total_profile,dark_profile,gas_profile,type2_profile,type3_profile,type4_profile])
	all_profiles = all_profiles[mask]

	#-------------------------------------------------------------------------
	if do_1e6:

		if sim[:-3]+'1e6' in models:
			f = h5py.File(d.datdir+'massprofiles_'+sim[:-3]+'1e6.hdf5','r')
			print(d.datdir+'massprofiles_'+sim[:-3]+'1e6.hdf5')
			hrfile = sim[:-3]+'1e6'
			hrfile = models[np.where(models==hrfile)[0][0]]

		elif sim.split('/')[1][:-3]+'1e6' in models:
			f = h5py.File(d.datdir+'massprofiles_'+sim.split('/')[1][:-3]+'1e6.hdf5','r')
			print(d.datdir+'massprofiles_'+sim.split('/')[1][:-3]+'1e6.hdf5')
			hrfile = sim.split('/')[1][:-3]+'1e6'
			hrfile = models_label[np.where(models==hrfile)[0][0]]

		else:
			f = h5py.File(d.datdir+'massprofiles_fiducial_1e6.hdf5','r')
			print(d.datdir+'massprofiles_fiducial_1e6.hdf5')
			hrfile = 'fiducial 1e6'

		max_snap = len(np.array(f['gas'])) - 1
		drange_6 = np.array(f['drange'])

		if snapnum > max_snap:
			gas_profile_6 = np.array(f['gas'])[max_snap]
			dark_profile_6 = np.array(f['dark'])[max_snap]
			type2_profile_6 = np.array(f['type2'])[max_snap]
			type3_profile_6 = np.array(f['type3'])[max_snap]
			type4_profile_6 = np.array(f['type4'])[max_snap]
		else:
			gas_profile_6 = np.array(f['gas'])[snapnum]
			dark_profile_6 = np.array(f['dark'])[snapnum]
			type2_profile_6 = np.array(f['type2'])[snapnum]
			type3_profile_6 = np.array(f['type3'])[snapnum]
			type4_profile_6 = np.array(f['type4'])[snapnum]

		if plot_initial:
			dark_profile_0_6 = np.array(f['dark'])[0]
		f.close()

		vols_6 = 4./3.*np.pi*(drange_6**3)
		total_profile_6 = gas_profile_6 + dark_profile_6 + type2_profile_6 + type3_profile_6 + type4_profile_6

		all_profiles_6 = np.array([total_profile_6,dark_profile_6,gas_profile_6,type2_profile_6,type3_profile_6,type4_profile_6])
		all_profiles_6 = all_profiles_6[mask]

	#-------------------------------------------------------------------------
	for j in range(len(all_types)):
		text = ax.annotate(all_types[j],xy=(0.05,0.3-(0.04*j)),xycoords='axes fraction',fontsize=11,color=all_colors[j])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=all_colors[j], edgecolor='white'), path_effects.Normal()])

		if do_density:
			rho_c = 3*(h*100)**2/(8*np.pi*m.G) * m.to_Msun_kpc3
			density = all_profiles[j]/vols
			ax.plot(drange,density,'-',c=all_colors[j],lw=1.5)

			if all_types[j]=='dark matter':
				rvir = drange[(density >= 200*rho_c)][-1]

				if plot_initial:
					density_0 = dark_profile_0/vols
					ax.plot(drange,density_0,'--',c='black',label='snapshot 000')

					rho_ratio = density_0 / density
					sel_in_2 = (rho_ratio > 1.4) & (rho_ratio < 1.8)
					is_2 = np.count_nonzero(sel_in_2)

					if is_2==0:
						core_radius=0
					else:
						ind = np.where(sel_in_2)[0][-1]
						core_radius = drange[ind]

					ax.axvline(x=core_radius,c='purple',alpha=0.6,label='core radius')

				if dicintio:
					snapfile = d.smuggledir + sim + '/snapshot_'+str(snapnum).zfill(3)

					darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
					type2mass = snapHDF5.read_block(snapfile, 'MASS', parttype=2)*(1.e10)/h
					type3mass = snapHDF5.read_block(snapfile, 'MASS', parttype=3)*(1.e10)/h
					type4mass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h

					Mstar = np.sum(type2mass) + np.sum(type3mass) + np.sum(type4mass)
					Mhalo = np.sum(darkmass)
					r_s = rvir / 15.

					#-----------------------------------------------------
					rho_s_range = np.logspace(3,10,200)

					difference = np.array([])
					for rho_s in rho_s_range:
						attempt = m.dicintio_profile(rho_s,r_s,drange,Mstar,Mhalo)
						difference = np.append(difference,np.abs(np.sum(attempt - density)))

					best_rho_s = rho_s_range[np.argmin(difference)]

					#-----------------------------------------------------
					rho_s_range = np.logspace(np.log10(best_rho_s/10),np.log10(best_rho_s*10),200)

					difference = np.array([])
					for rho_s in rho_s_range:
						attempt = m.dicintio_profile(rho_s,r_s,drange,Mstar,Mhalo)
						difference = np.append(difference,np.abs(np.sum(attempt - density)))

					best_rho_s = rho_s_range[np.argmin(difference)]
					
					#-----------------------------------------------------
					ax.plot(drange,m.dicintio_profile(best_rho_s,r_s,drange,Mstar,Mhalo),':',lw=1.5,label='Di Cintio Fit',c='red')

				if do_slope:
					x = drange
					y = np.log10(density)
					slope = np.array([])

					for q in range(len(drange)-1):
						dydx = (y[q+1] - y[q]) / (x[q+1] - x[q])
						slope = np.append(slope,dydx)


					ax2 = ax.twinx()
					ax2.plot(drange[:-1],slope,c='blue')
					text = ax.annotate(r'$\frac{d\log{\rho}}{dr}$',xy=(0.6,0.82),xycoords='axes fraction',fontsize=15,color='blue')
					text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=all_colors[j], edgecolor='white'), path_effects.Normal()])

			if do_1e6:
				# print('plotting '+all_types[j]+' 1e6')
				density_6 = all_profiles_6[j]/vols_6
				ax.plot(drange_6,density_6,'--',c=all_colors[j])



		elif do_velocity:
			ax.plot(drange,np.sqrt(m.Gprime*all_profiles[j]/drange),c=all_colors[j],lw=1.4)#,ls=styles[i])

		else:
			ax.plot(drange,all_profiles[j],c=all_colors[j])#,ls=styles[i])
	
	#-------------------------------------------------------------------------
	this_label = models_label[np.where(models==sim)[0][0]]

	if do_density:
		# ax.legend(loc='upper right',frameon=False,prop={'size':11})
		fname = 'rho_'+thismassfile+'_'
		ax.set_ylabel(r'density [M$_\odot$ kpc$^{-3}$]')
		ax.set_ylim(1e0,3e9)
		text = ax.annotate(this_label+', snapshot '+str(snapnum).zfill(3),xy=(0.45,0.95),xycoords='axes fraction',fontsize=11,color='black')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=all_colors[j], edgecolor='white'), path_effects.Normal()])

		if do_1e6:
			text = ax.annotate(hrfile+', snapshot '+str(max_snap).zfill(3)+' --',xy=(0.45,0.91),xycoords='axes fraction',fontsize=11,color='black')
			text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=all_colors[j], edgecolor='white'), path_effects.Normal()])
			fname = fname + '1e6_' 


	elif do_velocity:
		fname = 'vcirc_'+savenames[i]+'_'
		ax.set_ylabel(r'$v_\mathregular{circ}$ [km s$^{-1}$]')
		ax.set_ylim(1,100)
		leg = ax.legend(loc='upper right',frameon=True,fancybox=False,prop={'size':11})#,edgecolor='grey')

		frame = leg.get_frame()
		frame.set_color('white')
		frame.set_edgecolor('grey')
		frame.set_alpha(1)

		for j in range(len(all_types)):
			ax.annotate(all_types[j],xy=(0.05,0.95-(j*0.03)),xycoords='axes fraction',fontsize=11,color=all_colors[j])		

		text=ax.annotate(models_label[i],xy=(0.35,0.1),xycoords='axes fraction',fontsize=11,color='k')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='k', edgecolor='white'), path_effects.Normal()])
		text=ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.35,0.05),xycoords='axes fraction',fontsize=11,color='k')	
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='k', edgecolor='white'), path_effects.Normal()])
	
	else:
		fname = 'mass_'+sim+'_'
		ax.set_ylabel(r'mass [M$_\odot$]')
		ax.set_ylim(1e4,1e11)
		ax.legend(loc='lower right',frameon=False,prop={'size':11})

		for j in range(len(all_types)):
			ax.annotate(all_types[j],xy=(0.05,0.95-(j*0.03)),xycoords='axes fraction',fontsize=11,color=all_colors[j])		
	
	ax.set_xscale('log')
	ax.set_xlabel('distance [kpc]')
	ax.set_xlim(1e-1,1e2)
	ax.set_yscale('log')
	
	p.finalize(fig,fname+str(snapnum).zfill(3),save=1)

def plotradius(ptypes=np.array(['core','star']),do_bins=False):
	fig,ax = p.makefig(1,figx=7)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# print(ptypes)

	for sim in models:
		print(sim)
		i = np.where(models==sim)[0][0]

		f = h5py.File(d.datdir+'massprofiles_'+savenames[i]+'.hdf5','r')
		drange = np.array(f['drange'])
		dark_profile_all = np.array(f['dark'])

		max_snap = dark_profile_all.shape[0]

		core_radius = np.array([])
		star_radius = np.array([])
		time = np.array([])

		for snapnum in range(1,max_snap):
			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			
			header = snapHDF5.snapshot_header(snapfile)
			time = np.append(time, header.time)

			if 'core' in ptypes:
				# print('calculating core radius')
				dark_profile = dark_profile_all[snapnum]
				dark_profile_0 = dark_profile_all[0]
				vols = 4./3.*np.pi*(drange**3)
				# density = dark_profile/vols
				# density_0 = dark_profile_0/vols				
				# rho_ratio = density_0 / density
				# sel_in_2 = (rho_ratio > 1.4) & (rho_ratio < 1.8)
				# is_2 = np.count_nonzero(sel_in_2)

				avgmassenc = np.array([])
				avgmassenc_0 = np.array([])

				# for k in range(len(dark_profile)):
				# 	dark_profile = 


				if is_2==0:
					core_radius = np.append(core_radius,0.)
				else:
					ind = np.where(sel_in_2)[0][-1]
					core_radius = np.append(core_radius,drange[ind])

			if 'star' in ptypes:
				# print('calculating star radius')
				star_profile = np.array(f['type4'])[snapnum]
				total_star_mass = np.sum(snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h)

				if total_star_mass==0:
					star_radius = np.append(star_radius,0)	
				else:
					fractional_profile = star_profile / total_star_mass
					half_index = np.argmin(np.abs(fractional_profile-0.5))
					star_radius = np.append(star_radius,drange[half_index])

			elif ptype=='ratio':
				dark_profile = np.array(f['dark'])[snapnum]
				dark_profile_0 = np.array(f['dark'])[0]
				vols = 4./3.*np.pi*(drange**3)
				density = dark_profile/vols
				density_0 = dark_profile_0/vols
				
				rho_ratio = density_0 / density
				sel_in_2 = (rho_ratio > 1.4) & (rho_ratio < 1.8)
				is_2 = np.count_nonzero(sel_in_2)

				if is_2==0:
					core_radius = np.append(core_radius,0.)
				else:
					ind = np.where(sel_in_2)[0][-1]
					core_radius = np.append(core_radius,drange[ind])

				star_profile = np.array(f['type4'])[snapnum]
				total_star_mass = np.sum(snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h)

				if total_star_mass==0:
					star_radius = np.append(star_radius,0)	
				else:
					fractional_profile = star_profile / total_star_mass
					half_index = np.argmin(np.abs(fractional_profile-0.5))
					star_radius = np.append(star_radius,drange[half_index])

		f.close()

		if 'core' in ptypes:

			if do_bins:
				timebins = np.linspace(0, 2, 30)
				binwidth = (timebins[1] - timebins[0])/2
				rcore_mean = np.array([])

				for j in range(len(timebins)-1):
					leftbin = timebins[j]
					rightbin = timebins[j+1]

					sel = (time > leftbin) & (time < rightbin)
					rcore_mean = np.append(rcore_mean, np.mean(core_radius[sel]))

				ax.plot(timebins[0:-1]+binwidth,rcore_mean,color=colors_list[i],lw=1.6,alpha=1,ls='-')
				# ax.annotate(models_label[i],xy=(0.05,1-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])

			else:
				# print('plotting core radius')
				ax.plot(time,core_radius,colors_list[i],lw=1.6,ls='-',alpha=1)
			
			ax.set_ylabel(r'radius [kpc]')
			fname = 'radius_core+star_'+whichsims
			ax.annotate(models_label[i]+' ',xy=(0.7,0.95-(0.04*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])
			
			

			# ax.axvline(x=0.90)	
			# ax.axvline(x=1.08)	

		if 'star' in ptypes:
			# print('calculating core radius')
			ax.plot(time,star_radius,colors_list[i],ls='--',lw=1,alpha=0.7)
			
			ax.set_ylim(0,4)
			# ax.set_ylabel(r'$r_\mathregular{50\ast}$ [kpc]')
			# fname = 'radius_star_'+whichsims
			# ax.set_ylim(0,5.5)
			# ax.set_yticks([0,1,2,3,4,5])
			# ax.set_yticks([0.5,1.5,2.5,3.5,4.5,5.5],minor=True)
			# ax.annotate(models_label[i],xy=(0.65,0.98-(0.04*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])
		
		# elif ptype=='ratio':
		# 	ax.plot(time[5:],core_radius[5:]/star_radius[5:],colors_list[i],lw=1.6,alpha=0.7)
		# 	ax.set_ylabel(r'$r_\mathregular{core}/r_\mathregular{50\ast}$ [kpc]')
		# 	fname = 'radius_frac_'+whichsims
		# 	ax.set_ylim(0,0.8)
		# 	ax.annotate(models_label[i],xy=(0.05,0.98-(0.04*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])
	if 'core' in ptypes:
	# 	ax.plot(1e10,1e10,'-',c='k',label=r'$r_\mathregular{core}$')
		ax.annotate(r'solid: $r_\mathregular{core}$',xy=(0.7,0.9-(0.04*(i+1))),xycoords='axes fraction',fontsize=11,color='k')
	if 'star' in ptypes:
	# 	ax.plot(1e10,1e10,'--',c='k',label=r'$r_\mathregular{h\ast}$')
		ax.annotate(r'dashed: $r_\mathregular{h\ast}$',xy=(0.7,0.9-(0.04*(i+2))),xycoords='axes fraction',fontsize=11,color='k')

	ax.legend(frameon=False,loc='upper right',prop={'size':11})
	ax.set_xlabel(r'time [$h^{-1}$ Gyr]')
	ax.set_xlim(0,2)
	

	p.finalize(fig,fname,save=1)

def radius_alltypes(simnum):
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	# ax.plot(1e20,1e20,'k',ls='-',label='snapshot '+str(snapnum).zfill(3))

	snapdir = '/mainvol/ejahn/smuggle/output/'
	# snapdir = '/mainvol/ejahn/smuggle/output/live_halo_09.2019/'

	plot_types = np.array([0,2,3,4])
	all_types = np.array(['gas','type2','type3','type4'])
	colors = np.array(['blue','green','orange','red'])

	sims = np.array(['live_halo_02.2020/fiducial_1e5',
					 'live_halo_02.2020/compact_dwarf/fiducial_1e5',
					 'live_halo_02.2020/vareff_1e5',
					 'live_halo_09.2019/SFhigh_1e5'])
	sim = sims[simnum]
	print(sim)
	massfiles = np.array(['fiducial_1e5','cd_fiducial_1e5','vareff_1e5','SFhigh_1e5'])
	labels = np.array(['fiducial (ff) 1e5','compact dwarf, ff 1e5','variable efficiency 1e5','100% efficiency 1e5'])
	time = np.array([])
	f = h5py.File(d.datdir+'massprofiles_'+massfiles[simnum]+'.hdf5','r')
	drange = np.array(f['drange'])

	for j in range(len(all_types)):
		
		print(all_types[j])
		radius = np.array([])
		
		for snapnum in range(201):
			snapfile = snapdir+sim+'/snapshot_'+str(snapnum).zfill(3)
			profile = np.array(f[all_types[j]])[snapnum]
			total_mass = np.sum(snapHDF5.read_block(snapfile, 'MASS', parttype=plot_types[j])*(1.e10)/h)
			if j==0:
				header = snapHDF5.snapshot_header(snapfile)
				time = np.append(time, header.time)

			if total_mass==0:
				radius = np.append(radius,0)	
			else:
				fractional_profile = profile / total_mass
				half_index = np.argmin(np.abs(fractional_profile-0.5))
				radius = np.append(radius,drange[half_index])

		ax.plot(time,radius,colors[j],lw=1.6,alpha=0.7)
		ax.annotate(all_types[j],xy=(0.05,0.98-(0.04*(j+1))),xycoords='axes fraction',fontsize=11,color=colors[j])
	
	f.close()		

	ax.set_xlim(0,1)
	ax.set_xlabel(r'time [$h^{-1}$ Gyr]')

	ax.set_ylim(0.1,35)
	ax.set_yscale('log')
	ax.set_ylabel('half mass radius [kpc]')
	
	ax.set_title(massfiles[simnum]+'   ',loc='right',size=11)

	p.finalize(fig,'radius_alltypes_'+massfiles[simnum],save=1)
	print('')

def mass_v_time(dcut=5):
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	

	for sim in models:
		print(sim)
		i = np.where(models==sim)[0][0]
		time = np.array([])

		if 'compact_dwarf' in sim:
			thismassfile = 'cd_' + sim.split('/')[-1]
		else:
			thismassfile = sim

		f = h5py.File(d.datdir+'massprofiles_'+thismassfile+'.hdf5','r')
		drange = np.array(f['drange'])
		d_index = m.find_nearest(drange,5,getindex=True)

		gas_mass = np.array([])
		type4_mass = np.array([])

		max_snap = len(np.array(f['gas']))

		for snapnum in range(1,max_snap):
			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)


			header = snapHDF5.snapshot_header(snapfile)
			time = np.append(time, header.time)

			gas_profile = np.array(f['gas'])[snapnum]
			type4_profile = np.array(f['type4'])[snapnum]

			gas_mass = np.append(gas_mass,gas_profile[d_index])
			type4_mass = np.append(type4_mass,type4_profile[d_index])

		ax.plot(time,gas_mass,'--',color=colors_list[i],lw=1.5)
		ax.plot(time,type4_mass,'-',color=colors_list[i],lw=1.5)
		ax.annotate(models_label[i],xy=(0.15,0.05+(0.04*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])

	ax.plot(3,1e12,'--',lw=1.5,c='k',label='gas mass')
	ax.plot(3,1e12,'-',lw=1.5,c='k',label='type 4 mass')

	ax.legend(frameon=False,loc='lower right',prop={'size':11})

	ax.set_xlabel(r'time [$h^{-1}$ Gyr]')
	ax.set_xticks([0,0.5,1,1.5,2])
	ax.set_xticks([0.25,0.75,1.25,1.75],minor=True)
	ax.set_xlim(0,2)

	ax.set_yscale('log')
	ax.set_ylim(1e6,1e9)
	ax.set_ylabel('Mass < '+str(dcut)+r' kpc [M$_\odot$]')

	p.finalize(fig,'Min'+str(dcut)+'time_'+whichsims,save=1)

def SFH():
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	# ax.plot(1e20,1e20,'k',ls='-',label='snapshot '+str(snapnum).zfill(3))

	for sim in models:
		print(sim)
		i = np.where(models==sim)[0][0]

		snapfile = d.smuggledir+sim+'/snapshot_400'

		ages = snapHDF5.read_block(snapfile,'AGE ',parttype=4)

		ax.hist(ages,bins=20,color=colors_list[i],histtype='step',normed=True)
		ax.hist(ages,bins=20,color=colors_list[i],histtype='stepfilled',alpha=0.4,normed=True)
		# hist, bin_edges = np.histogram(ages,bins=20,density=True)
		# binwidth = bin_edges[1]-bin_edges[0]
		# ax.plot(bin_edges[:-1]+binwidth,hist,c=colors_list[i])
		if whichsims=='fixISM':
			ax.annotate(models_label[i],xy=(0.60,1-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])
		else:
			ax.annotate(models_label[i],xy=(0.05,1-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])


	ax.set_xlabel(r'$t_\mathregular{form}$ (Gyr; Type 4)')
	# ax.set_yscale('log')
	ax.set_ylim(0.2,1.2)
	ax.set_ylabel('frequency')
	p.finalize(fig,'SFH_'+whichsims,save=1)

def outflow_velocity(sim,snapnum,vtype,dist,width,save=True,exclude=False):
	if not(sim in models):
		raise ValueError('please choose a sim from models')
	i = np.where(models==sim)[0][0]

	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
	thisname = 'outflow_movie_'+savenames[i]+'_'+vtype

	inner = dist - width/2
	outer = dist + width/2

	darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
	dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
	x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
	y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
	z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
	dark_cm = np.array([x_cm,y_cm,z_cm]).T

	gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
	gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
	gas_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=0)
	
	if exclude:
		sel_exc = (np.abs(gas_pos[:,0]) > 5) & (np.abs(gas_pos[:,1]) > 5) & (np.abs(gas_pos[:,2]) > 1) 
		gas_pos = gas_pos[sel_exc]
		gas_vel = gas_vel[sel_exc]

		text = ax.annotate(r'excluded [5,5,1]',xy=(0.65,0.8),xycoords='axes fraction',fontsize=11,color='k')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
			edgecolor='white'), path_effects.Normal()])

		thisname += '_excludebox'

	d_gas  = np.linalg.norm(gas_pos-dark_cm, axis=1)
	gas_vel = gas_vel[(d_gas > inner) & (d_gas < outer)]
	
	if vtype=='vr':
		gas_rhat = ((gas_pos - dark_cm).T / d_gas).T
		gas_rhat = gas_rhat[(d_gas > inner) & (d_gas < outer)]

		gas_vr = np.array([])
		for j in range(len(gas_vel)):
			gas_vr = np.append(gas_vr,np.dot(gas_vel[j],gas_rhat[j]))

		ax.hist(gas_vr,bins=20,color=colors_list[i],histtype='step',normed=True)
		ax.hist(gas_vr,bins=20,color=colors_list[i],histtype='stepfilled',alpha=0.4,normed=True)
		text = ax.annotate(r'$n$ = '+str(len(gas_vr)),xy=(0.7,0.85),xycoords='axes fraction',fontsize=11,color='k')

		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
			edgecolor='white'), path_effects.Normal()])

		ax.set_xlabel(r'gas radial velocity $v_r$ [km/s]')

	elif vtype=='vz':
		gas_vz = gas_vel[:,2]

		ax.hist(gas_vz,bins=20,color=colors_list[i],histtype='step',normed=True)
		ax.hist(gas_vz,bins=20,color=colors_list[i],histtype='stepfilled',alpha=0.4,normed=True)
		text = ax.annotate(r'$n$ = '+str(len(gas_vz)),xy=(0.7,0.85),xycoords='axes fraction',fontsize=11,color='k')

		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

		ax.set_xlabel(r'gas vertical velocity $v_z$ [km/s, gas]')


	text = ax.annotate(models_label[i]+'\nsnapshot '+str(snapnum).zfill(3),
		xy=(0.05,0.9),xycoords='axes fraction',fontsize=11,color=colors_list[i])
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])

	text = ax.annotate(str(inner)+r' kpc < $d_\mathregular{gas}$ < '+str(outer)+' kpc',
		xy=(0.5,0.9),xycoords='axes fraction',fontsize=11,color='k')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', 
		edgecolor='white'), path_effects.Normal()])


	# ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False, 
	# 	direction='in',labelsize=16,length=3)


	ax.set_ylim(0,0.04)
	ax.set_yticks([0,0.01,0.02,0.03,0.04])
	ax.set_yticklabels(['','','','',''])
	ax.set_yticks([0.005,0.015,0.025,0.035],minor=True)

	ax.set_xlim(-200,200)
	ax.set_xticks([-200,-150,-100,-50,0,50,100,150,200])
	ax.set_xticklabels(['-200','','-100','','0','','100','','200'])
	ax.set_xticks([-175,-125,-75,-25,25,75,125,175],minor=True)

	ax.set_ylabel('freq [arbitrary units]')

	thisname += '/'+str(inner)+'-'+str(outer)+'_'+str(snapnum).zfill(3)+'.png'

	# p.finalize(fig,thisname,save=save)
	print('saving figure: /home/ejahn003/movie_frames/'+thisname)
	plt.savefig('/home/ejahn003/movie_frames/'+thisname,format='png',dpi=200)
	# plt.show()

def alpha_proj(sim,snapnum,bound=10,do_epsilon=True,do_hist=True):
	if not(sim in models):
		raise ValueError('please choose one of the available simulations')

	i = np.where(models==sim)[0][0]

	fig,ax = p.makefig(1,figx=8,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	if not(do_hist):
		plt.style.use('dark_background')

	#create mass space for colormap to sample
	# c = np.linspace(1e-3,1e7,1000)
	# c = np.logspace(-3,0,1000)
	c = np.linspace(-2,0)
	norm = mplcolors.Normalize(vmin=c.min(), vmax=c.max())
	cmap = cm.ScalarMappable(norm=norm, cmap=cm.plasma_r)
	cmap.set_array([])

	#----------------------------------------------------------------------------------------------
	snapfile = outdirs[i]+sim+'/snapshot_'+str(snapnum).zfill(3)
	# print(snapfile)

	darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
	dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

	x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
	y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
	z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
	dark_cm = np.array([x_cm,y_cm,z_cm]).T

	gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
	gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h - dark_cm
	

	if do_epsilon:
		gas_rho = snapHDF5.read_block(snapfile, 'RHO ', parttype=0)*1e10 #Msun / kpc^3
		gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0) #Msun / year

		gas_rho = gas_rho*6.768e-32 #now in cgs (g/cm^3)
		G_cgs = 6.6743e-8 #cm^3 / (g s^2)


		tdyn = np.sqrt(3*np.pi/(32*G_cgs*gas_rho)) #should be in seconds
		tdyn = tdyn / (3.154e7) #should now be in years


		# sel_0 = (gas_rho > 0)
		# gas_rho = gas_rho[sel_0]
		# gas_sfr = gas_sfr[sel_0]
		# gasmass = gasmass[sel_0]
		# gas_pos = gas_pos[sel_0]

		#tdyn = np.sqrt(3*np.pi/(32*m.Gsim*gas_rho))
		# rinf = np.logical_not(np.isnan(tdyn)) & (tdyn < np.inf)
		# tdyn = tdyn[rinf]
		# gas_sfr = gas_sfr[rinf]
		# gasmass = gasmass[rinf]
		# gas_pos = gas_pos[rinf]

		epsilon = (gas_sfr/gasmass)*tdyn

		# epsilon = epsilon[(epsilon > 0)]
		# gas_pos = gas_pos[(epsilon > 0)]

		# print(len(epsilon))
		# print(np.amin(epsilon))
		# print(np.amax(epsilon))

	else:
		gas_virial = snapHDF5.read_block(snapfile, 'ALPH', parttype=0)
	#----------------------------------------------------------------------------------------------
	if do_hist:
		fname='snapshot_'+str(snapnum).zfill(3)
		sel = np.logical_not(np.isnan(epsilon)) & (epsilon != 0)

		if len(epsilon[sel])==0:
			return

		ax.hist(epsilon[sel],histtype='step',normed=True,bins=30,log=True,lw=2,color='darkturquoise')
		ax.hist(epsilon[sel],histtype='stepfilled',normed=True,bins=30,log=True,lw=0,color='darkturquoise',alpha=0.3)
		
		ax.annotate('n='+str(len(epsilon[sel])),xy=(0.65,0.95),xycoords='axes fraction',fontsize=11,color='black')

		ax.set_xlim(1e-2,1)
		ax.set_xlabel(r'SF efficiency, $\varepsilon$')
		ax.set_xscale('log')

		ax.set_ylabel('frequency')
		ax.set_ylim(1e-1,1e2)
		

	else:
		# xbins = np.linspace(-bound,bound,200)
		# ybins = np.linspace(-bound,bound,200)
		# bw = (xbins[1] - xbins[0])/2

		# total = len(xbins)-1
		
		# allvalues = np.array([])

		# for j in range(total):
		# 	selx = (gas_pos[:,0] > xbins[j]) & (gas_pos[:,0] < xbins[j+1])
		# 	thisx = xbins[j]+bw

		# 	printthing = str(np.round(j/2,0))+'%'
		# 	# print(np.int(j/(total-1)*columns))/

		# 	# printthing = '['+('-'*np.int(j/(total-1)*(columns-2))).ljust(columns-2,' ')+']'
		# 	sys.stdout.write(printthing)
		# 	sys.stdout.flush()
		# 	sys.stdout.write("\b" * (len(printthing)))

		# 	for k in range(len(ybins)-1):
		# 		sely = (gas_pos[:,1] > ybins[k]) & (gas_pos[:,1] < ybins[k+1])
		# 		thisy = ybins[k]+bw

		# 		this_sel = selx & sely & (gas_pos[:,2] > -1) & (gas_pos[:,2] < 1)

		# 		if (np.count_nonzero(this_sel) > 0):
		# 			if do_epsilon:
		# 				meanvalue = np.mean(epsilon[this_sel])
		# 				if meanvalue > 0:
		# 					ax.plot(thisx,thisy,'o',ms=2,mew=0,c=cmap.to_rgba(m.find_nearest(c,np.log10(meanvalue))))

		# 			else:
		# 				meanvalue = np.mean(gas_virial[this_sel])
		# 				ax.plot(thisx,thisy,'o',ms=2,mew=0,c=cmap.to_rgba(m.find_nearest(c,np.log10(meanvalue))))
		epsilon = epsilon[(epsilon != 0)]
		gas_pos = gas_pos[(epsilon != 0)]
		print(epsilon.shape)
		print(gas_pos.shape)
		for j in range(len(epsilon)):
			ax.plot(gas_pos[j,0],gas_pos[j,1],'o',ms=2,mew=0,c=cmap.to_rgba(m.find_nearest(c,np.log10(epsilon[j]))))


		#----------------------------------------------------------------------------------------------
		if do_epsilon:
			fname = 'epsilon_proj_'+savenames[i]+'_'+str(snapnum).zfill(3)
			fig.colorbar(cmap,label=r'log$_{10}(\varepsilon)$')
		else:
			fname = 'alpha_proj_'+savenames[i]+'_'+str(snapnum).zfill(3)
			fig.colorbar(cmap,label=r'$\alpha$')

		ax.set_xlim(-bound,bound)
		ax.set_xlabel('x [kpc]')
		ax.set_ylim(-bound,bound)
		ax.set_ylabel('y [kpc]')

	ax.annotate(models_label[i],xy=(0.05,0.95),xycoords='axes fraction',fontsize=11,color='black')
	ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.05,0.9),xycoords='axes fraction',fontsize=11,color='black')

	# p.finalize(fig,fname,save=0)
	sd = '/home/ejahn003/movie_frames/epsilon_hist_vareff_v2/'
	print('saving figure: '+fname+'.png')
	plt.savefig(sd+fname+'.png',format='png',dpi=200)

def dicintio_test():
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	X_array = np.array([-4.5,-4,-3.5,-3,-2.5,-2,-1.5])
	rho_s_array = np.array([1e1,1e2,1e3,1e4,1e5,1e6,1e7])
	drange = np.logspace(-2,2,200)

	colors = np.array(['black','darkviolet','red','orange','yellowgreen','darkturquoise','blue','firebrick','violet','pink'])	

	f = h5py.File(d.datdir+'massprofiles_fiducial_1e5.hdf5','r')
	drange_rv = np.array(f['drange'])
	mass = np.array(f['dark'])[200]
	f.close()

	vols = 4./3.*np.pi*(drange_rv**3)
	density = mass / vols

	rho_c = 3*(h*100)**2/(8*np.pi*m.G) * m.to_Msun_kpc3
	rvir = drange_rv[(density >= 200*rho_c)][-1]
	r_s = rvir / 15

	for X in X_array:
		i = np.where(X_array==X)[0][0]
		rho_s = rho_s_array[i]

		alpha = 2.94 - np.log10(	( 10**(X+2.33) )**-1.08 	+ 	( 10**(X+2.33) )**2.29		)
		beta = 4.23 + 1.34*X + 0.26*X*X
		gamma = -0.06 + np.log10(	( 10**(X+2.56) )**-0.68		+	  10**(X+2.56)				)

		profile = rho_s / ( (drange/r_s)**gamma  *  (1 + (drange/r_s)**alpha)**((beta-gamma)/alpha) )

		ax.plot(drange,profile,'-',lw=2,c=colors[i])
		ax.annotate('X = '+str(X),xy=(0.05,0.05+(0.045*i)),xycoords='axes fraction',fontsize=12,color=colors[i])

	
	ax.annotate('Di Cintio Curves',xy=(0.5,0.95),xycoords='axes fraction',fontsize=12,color='k')	
	equation = r'$\rho(r) = \frac{\rho_s}{(\frac{r}{r_s})^\gamma [1+(\frac{r}{r_s})^\alpha]^{(\beta-\gamma)/\alpha}}$'
	ax.annotate(equation,xy=(0.45,0.86),xycoords='axes fraction',fontsize=20,color='k')	

	ax.annotate(r'X = log$_{10}$(M$_\ast$ / M$_\mathregular{halo}$)',xy=(0.25,0.25),xycoords='axes fraction',fontsize=12,color='k')

	ax.set_xlim(1e-2,1e2)
	ax.set_xscale('log')
	ax.set_xlabel('distance [kpc]')
		
	ax.set_ylim(1e-4,1e10)
	ax.set_yscale('log')
	ax.set_ylabel('density [arbitrary]')
	ax.set_yticks([])
	ax.set_yticklabels([])

	p.finalize(fig,save=1)

def timescales(sim,snapnum,whichy,whichx):
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	if not(sim in models):
		raise ValueError('please choose sim in models')

	i = np.where(models == sim)[0][0]

	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
	print(snapfile)

	darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
	dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h

	x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
	y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
	z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
	dark_cm = np.array([x_cm,y_cm,z_cm]).T

	gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
	gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h - dark_cm
	
	gas_rho = snapHDF5.read_block(snapfile, 'RHO ', parttype=0) #Msun / kpc^3
	gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0) / (3.154e7) # converted to Msun / sec

	tdyn = np.sqrt(3*np.pi/(32*m.Gsim*gas_rho))
	rinf = np.logical_not(np.isnan(tdyn)) & (tdyn < np.inf)

	# tdyn = tdyn[rinf]
	# gas_sfr = gas_sfr[rinf]
	# gasmass = gasmass[rinf]
	# gas_pos = gas_pos[rinf]
	
	# epsilon =gas_sfr/gasmass*tdyn

	if whichy=='tdyn':
		yplt = tdyn[rinf]
		ax.set_yscale('log')
		ax.set_ylabel(r'$t_\mathregular{dyn}$ [sec]')
	elif whichy=='tsfr':
		yplt = gasmass[rinf] / gas_sfr[rinf]
		ax.set_yscale('log')
		ax.set_ylabel(r'$t_\mathregular{sfr}$ [sec]')
	else:
		raise ValueError('unknown whichy')
	
	if whichx=='distance':
		xplt = np.linalg.norm(gas_pos[rinf],axis=1)
		ax.set_xlabel(r'dist [kpc]')
	elif whichx=='density':
		xplt = gas_rho[rinf]
		ax.set_xscale('log')
		ax.set_xlabel(r'$\rho_\mathregular{gas}$ [M$_\odot$ kpc$^{-3}$]')
	else:
		raise ValueError('unknown whichx')

	ax.plot(xplt,yplt,'o',mew=0,ms=2)
	ax.set_title(savenames[i],size=11,loc='right')

	figname = whichy+'_'+whichx+'_'+savenames[i]+'_'+str(snapnum).zfill(3)
	p.finalize(fig,figname,save=1)

def masstime():
	print('plotting type 4 mass versus time')
	# if not(sim in models):
	# 	raise ValueError('please use a sim in the list of models')
	fig,ax = p.makefig(1,figx=8,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	
	

	for sim in models:
		i = np.where(models==sim)[0][0]

		f = h5py.File(d.datdir+'massprofiles_'+savenames[i]+'.hdf5','r')
		type4_profile = np.array(f['type4'])
		f.close()

		type4mass = np.array([])
		all_time = np.array([])

		for num in range(len(type4_profile)):
			type4mass = np.append(type4mass,type4_profile[num][-1])

			all_time = np.append(all_time, snapHDF5.snapshot_header(d.smuggledir + sim + '/snapshot_' + str(num).zfill(3)).time)

		ax.plot(all_time,type4mass,lw=1.6,c=colors_list[i])
		ax.annotate(models_label[i],xy=(0.7,0.25-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])


	ax.set_yscale('log')
	ax.set_ylim(1e7,1.5*1e8)
	ax.set_ylabel(r'type 4 mass [M$_\odot$]')

	# ax.set_xscale('log')
	ax.set_xlim(0,2)
	ax.set_xticks([0,0.5,1,1.5,2])
	ax.set_xticks([0.25,0.75,1.25,1.75],minor=True)
	ax.set_xlabel('time [Gyr]')

	p.finalize(fig,'masstime_'+whichsims,save=1)

def Jz_profile(snapnum):
	fig,ax = p.makefig(1,figx=8,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for sim in models:
		i = np.where(models==sim)[0][0]

		snapfile = outdirs[i]+sim+'/snapshot_'+str(snapnum).zfill(3)
		print(sim)
		# print(snapfile)
		gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
		gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
		gas_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=0)

		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
		dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
		dark_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=1)

		x_cm = np.sum(dark_pos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(dark_pos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(dark_pos[:,2] * darkmass) / np.sum(darkmass)
		dark_cm = np.array([x_cm,y_cm,z_cm]).T

		vx_cm = np.sum(dark_vel[:,0] * darkmass) / np.sum(darkmass)
		vy_cm = np.sum(dark_vel[:,1] * darkmass) / np.sum(darkmass)
		vz_cm = np.sum(dark_vel[:,2] * darkmass) / np.sum(darkmass)
		dark_v_cm = np.array([vx_cm,vy_cm,vz_cm]).T

		gas_pos = gas_pos-dark_cm
		gas_vel = gas_vel-dark_v_cm

		r_gas = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2)
		d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)

		rho = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2)
		phi = np.arctan(gas_pos[:,0]/gas_pos[:,1])

		vphi = (gas_pos[:,0]*gas_vel[:,1] - gas_pos[:,1]*gas_vel[:,0])*(np.cos(phi) - np.sin(phi))/rho

		Jz = vphi#*gasmass

		drange = np.logspace(-1,2,10)
		Jz_median = np.array([])
		Jz_pos_error = np.array([])
		Jz_neg_error = np.array([])
		Jz_sim_error = np.array([])
		r_gas_median = np.array([])

		for j in range(len(drange)-1):
			sel = (r_gas > drange[j]) & (r_gas < drange[j+1])
			this_Jz = Jz[sel]
			Jz_median = np.append(Jz_median,np.median(Jz[sel]))
			sel_p = (this_Jz > np.median(this_Jz))
			sel_n = (this_Jz < np.median(this_Jz))
			this_Jz_p_er = np.std(this_Jz[sel_p])
			this_Jz_n_er = np.std(this_Jz[sel_n])

			Jz_pos_error = np.append(Jz_pos_error,this_Jz_p_er)
			Jz_neg_error = np.append(Jz_neg_error,this_Jz_n_er)

			r_gas_median = np.append(r_gas_median,np.median(r_gas[sel]))

		Jz_median = Jz_median[Jz_median>1]
		r_gas_median = r_gas_median[Jz_median>1]

		if colors_list[i]=='black':
			ax.plot(r_gas_median,Jz_median,'-o',c=colors_list[i],mec='white',lw=1.5,zorder=i*10+100)
		else:
			ax.plot(r_gas_median,Jz_median,'-o',c=colors_list[i],mec='black',lw=1.5,zorder=i*10+100)

		ax.plot(r_gas,Jz,'o',mew=0,mfc=colors_list[i],ms=1.5,alpha=0.05,zorder=i*10)
		text = ax.annotate(models_label[i],xy=(0.05,0.95-(0.05*i)),xycoords='axes fraction',fontsize=11,color=colors_list[i],zorder=i*10+1000)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=colors_list[i], 
			edgecolor='white'), path_effects.Normal()])

		#---plot-vcirc-----------------------------------------------------------------------------
		if not(sim=='ff_tiny_1e6'):
			f = h5py.File(d.datdir+'massprofiles_'+savenames[i]+'.hdf5','r')
			drange = np.array(f['drange'])
			gas_profile = np.array(f['gas'])[snapnum]
			dark_profile = np.array(f['dark'])[snapnum]
			type2_profile = np.array(f['type2'])[snapnum]
			type3_profile = np.array(f['type3'])[snapnum]
			type4_profile = np.array(f['type4'])[snapnum]
			f.close()

			total_profile = gas_profile + dark_profile + type2_profile + type3_profile + type4_profile
			ax.plot(drange,np.sqrt(m.Gprime*total_profile/drange),c=colors_list[i],lw=1,ls='--',zorder=i*10+1000)

	#---plotting-stuff-----------------------------------------------------------------------------
	ax.annotate('gas [type 0]',xy=(0.8,0.95),xycoords='axes fraction',fontsize=11,color='k')
	ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.8,0.9),xycoords='axes fraction',fontsize=11,color='k')
	ax.annotate(r'-- $v_\mathregular{circ}$',xy=(0.8,0.85),xycoords='axes fraction',fontsize=17,color='k')


	ax.set_xlim(0.1,100)
	ax.set_xscale('log')
	ax.set_xlabel('r [kpc]')

	ax.set_ylim(1e0,2e2)
	ax.set_yscale('log')
	ax.set_ylabel(r'$j_z$ [km s$^{-1}$]')

	p.finalize(fig,'jz_'+whichsims,save=1)

def list_rcores(sim):
	f = h5py.File(d.datdir+'massprofiles_'+savenames[np.where(models==sim)[0][0]]+'.hdf5','r')
	drange = np.array(f['drange'])
	dark_profile_all = np.array(f['dark'])
	f.close()

	max_snap = dark_profile_all.shape[0]

	core_radius = np.array([])
	star_radius = np.array([])
	time = np.array([])

	for snapnum in range(1,max_snap):
		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
		
		header = snapHDF5.snapshot_header(snapfile)
		time = np.append(time, header.time)

		dark_profile = dark_profile_all[snapnum]
		dark_profile_0 = dark_profile_all[0]
		vols = 4./3.*np.pi*(drange**3)
		density = dark_profile/vols
		density_0 = dark_profile_0/vols
		
		rho_ratio = density_0 / density
		sel_in_2 = (rho_ratio > 1.4) & (rho_ratio < 1.8)
		is_2 = np.count_nonzero(sel_in_2)

		if is_2==0:
			core_radius = np.append(core_radius,0.)
		else:
			ind = np.where(sel_in_2)[0][-1]
			core_radius = np.append(core_radius,drange[ind])

	imax = np.where(core_radius == np.amax(core_radius))[0][0]
	imin = np.where(core_radius == np.amin(core_radius))[0][0] 

	print('max core radius at snapshot '+str(imax).zfill(3))
	print('min core radius at snapshot '+str(imin).zfill(3))

	return imax,imin

def three_rho_plots(sim):
	i = np.where(models==sim)[0][0]
	# imax, imin = list_rcores(sim)
	fig, axarr = p.makefig(n_panels='3_horiz',figx=15,figy=5)

	f = h5py.File(d.datdir+'massprofiles_'+savenames[np.where(models==sim)[0][0]]+'.hdf5','r')
	drange = np.array(f['drange'])
	dark_mass_all = np.array(f['dark'])
	f.close()

	max_snap = dark_mass_all.shape[0]
	vols = 4./3.*np.pi*(drange**3)

	#---calculate-all-core-radius------------------------------------------------------------------

	core_radius = np.array([])
	star_radius = np.array([])
	time = np.array([])

	for snapnum in range(1,max_snap):
		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
		
		header = snapHDF5.snapshot_header(snapfile)
		time = np.append(time, header.time)

		dark_mass = dark_mass_all[snapnum]
		# dark_mass_0 = dark_mass_all[0]
		this_rho = dark_mass/vols
		# density_0 = dark_mass_0/vols
		dark_rho_0 = dark_mass_all[0]/vols

		rho_ratio = dark_rho_0 / this_rho
		sel_in_2 = (rho_ratio > 1.4) & (rho_ratio < 1.8)
		is_2 = np.count_nonzero(sel_in_2)

		if is_2==0:
			core_radius = np.append(core_radius,0.)
		else:
			ind = np.where(sel_in_2)[0][-1]
			core_radius = np.append(core_radius,drange[ind])

	imax = np.where(core_radius == np.amax(core_radius))[0][0]
	imin = np.where(core_radius == np.amin(core_radius))[0][0] 
	inb = m.find_nearest(core_radius,0.5*np.amax(core_radius),getindex=True)

	print('max core radius at snapshot '+str(imax).zfill(3))
	print('half max core radius at snapshot '+str(inb).zfill(3))
	print('min core radius at snapshot '+str(imin).zfill(3))

	#---plot-density-profiles----------------------------------------------------------------------

	dark_rho_0 = dark_mass_all[0]/vols
	dark_rho_max = dark_mass_all[imax]/vols
	dark_rho_min = dark_mass_all[imin]/vols

	# inb = np.random.choice(np.arange(50,max_snap))
	# while inb==imax:
	# 	inb = np.random.choice(np.arange(50,max_snap))

	dark_rho_inb = dark_mass_all[inb]/vols

	axarr[0].plot(drange,dark_rho_min,lw=1.5,c='k')
	axarr[1].plot(drange,dark_rho_inb,lw=1.5,c='k',label='current DM profile')
	axarr[2].plot(drange,dark_rho_max,lw=1.5,c='k')
	axarr[0].plot(drange,dark_rho_0,'--',lw=1,c='grey')
	axarr[1].plot(drange,dark_rho_0,'--',lw=1,c='grey',label='initial DM profile')
	axarr[2].plot(drange,dark_rho_0,'--',lw=1,c='grey')
	axarr[0].axvline(x=core_radius[imin],ls=':',lw=1.5,color='violet')
	axarr[1].axvline(x=core_radius[inb],ls=':',lw=1.5,color='violet',label=r'$r_\mathregular{core}$')
	axarr[2].axvline(x=core_radius[imax],ls=':',lw=1.5,color='violet')

	axarr[1].legend(loc='upper right',frameon=False,prop={'size':11})

	p.clear_axes(axarr[0])
	p.clear_axes(axarr[1])
	p.clear_axes(axarr[2])

	axarr[0].set_yscale('log')
	axarr[0].set_ylabel(r'$\rho_\mathregular{DM}$ [M$_\odot$ kpc$^{-3}$]')
	axarr[0].set_ylim(1e6,3e9)

	axarr[1].set_xlabel('distance [kpc]')

	axarr[0].set_xscale('log')
	axarr[0].set_xlim(1e-1,1e1)
	axarr[1].set_xscale('log')
	axarr[1].set_xlim(1e-1,1e1)
	axarr[2].set_xscale('log')
	axarr[2].set_xlim(1e-1,1e1)

	axarr[0].set_xticks([1e-1,1e0]);		axarr[0].set_xticklabels(['0.1','1'])
	axarr[1].set_xticks([1e-1,1e0]);		axarr[1].set_xticklabels(['0.1','1'])
	axarr[2].set_xticks([1e-1,1e0,1e1]);		axarr[2].set_xticklabels(['0.1','1','10'])

	axarr[0].annotate('snapshot '+str(imin).zfill(3),xy=(0.5,0.15),xycoords='axes fraction',fontsize=11,color='black')
	axarr[1].annotate('snapshot '+str(inb).zfill(3),xy=(0.5,0.15),xycoords='axes fraction',fontsize=11,color='black')
	axarr[2].annotate('snapshot '+str(imax).zfill(3),xy=(0.5,0.15),xycoords='axes fraction',fontsize=11,color='black')

	axarr[0].annotate(r'$r_\mathregular{core}$ = '+str(np.round(core_radius[imin],1))+' kpc',
		xy=(0.5,0.1),xycoords='axes fraction',fontsize=11,color='black')
	axarr[1].annotate(r'$r_\mathregular{core}$ = '+str(np.round(core_radius[inb],1))+' kpc',
		xy=(0.5,0.1),xycoords='axes fraction',fontsize=11,color='black')
	axarr[2].annotate(r'$r_\mathregular{core}$ = '+str(np.round(core_radius[imax],1))+' kpc',
		xy=(0.5,0.1),xycoords='axes fraction',fontsize=11,color='black')

	#		ax.annotate(models_label[i],xy=(0.05,1-(0.05*(i+1))),xycoords='axes fraction',fontsize=11,color=colors_list[i])

	p.finalize(fig,'triple_rho_'+savenames[i],save=1)

def newprofiles(type='rho'):
	fig,ax = p.makefig(1,figx=6.5,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	sims = np.array(['compact_dwarf/fiducial_1e5','fiducial_relaxed_1e5','fiducial_relaxed_sf_1e5'])
	labels = np.array(['cd ff 1e5','cd adiabatic 1e5','cd relaxed 1e5'])
	massprofiles = np.array(['cd_fiducial_1e5','cdffrlx_1e5','cdffrlx_sf_1e5'])
	colors = np.array(['black','blue','red'])
	snapnums = np.array([400,100,300])

	for sim in sims:
		i = np.where(sims==sim)[0][0]
		print(i,sim)

		f = h5py.File(d.datdir+'massprofiles_'+massprofiles[i]+'.hdf5','r')

		dark_profile = np.array(f['dark'])
		drange = np.array(f['drange'])
		gas_profile = np.array(f['gas'])
		type2_profile = np.array(f['type2'])
		type3_profile = np.array(f['type3'])
		type4_profile = np.array(f['type4'])

		total_profile = dark_profile[snapnums[i]] + gas_profile[snapnums[i]] + type2_profile[snapnums[i]] + type3_profile[snapnums[i]] + type4_profile[snapnums[i]]

		vols = 4./3.*np.pi*(drange**3)

		

		if type=='vcirc':
			ax.annotate(labels[i]+', snapshot '+str(snapnums[i]).zfill(3),xy=(0.05,0.95-(i*0.04)),xycoords='axes fraction',fontsize=11,color=colors[i])
			if i==0:
				ax.plot(drange,np.sqrt(m.Gprime*dark_profile[snapnums[i]]/drange),c=colors[i],ls='--',label='DM',lw=1.5)
				ax.plot(drange,np.sqrt(m.Gprime*total_profile/drange),c=colors[i],label='total',lw=1.5)
				initial_profile = dark_profile[0] + gas_profile[0] + type2_profile[0] + type3_profile[0] + type4_profile[0]
				ax.plot(drange,np.sqrt(m.Gprime*initial_profile/drange),c='grey',label='initial',lw=1.2)
			else:
				ax.plot(drange,np.sqrt(m.Gprime*dark_profile[snapnums[i]]/drange),c=colors[i],ls='--',lw=1.5)
				ax.plot(drange,np.sqrt(m.Gprime*total_profile/drange),c=colors[i],lw=1.5)	

		elif type=='rho':
			ax.annotate(labels[i]+', snapshot '+str(snapnums[i]).zfill(3),xy=(0.05,0.25-(i*0.04)),xycoords='axes fraction',fontsize=11,color=colors[i])
			if i==0:
				ax.plot(drange,dark_profile[snapnums[i]]/vols,c=colors[i],ls='--',label='DM',lw=1.5)
				ax.plot(drange,total_profile/vols,c=colors[i],label='total',lw=1.5)
				initial_profile = dark_profile[0] + gas_profile[0] + type2_profile[0] + type3_profile[0] + type4_profile[0]
				ax.plot(drange,initial_profile/vols,c='grey',label='initial',lw=1.2)
			else:
				ax.plot(drange,dark_profile[snapnums[i]]/vols,c=colors[i],ls='--',lw=1.5)
				ax.plot(drange,total_profile/vols,c=colors[i],lw=1.5)		

	if type=='rho':
		ax.set_xlim(0.1,10)
		ax.set_ylim(1e6,3e9)
		ax.set_yscale('log')
		ax.set_ylabel(r'$\rho$ [M$_\odot$ kpc$^{-3}$]')
		ax.legend(loc='upper right',frameon=False,prop={'size':11})
	elif type=='vcirc':
		ax.set_ylabel(r'$v_\mathregular{circ}$ [km s$^{-1}$]')
		ax.legend(loc='lower right',frameon=False,prop={'size':11})
		ax.set_xlim(0.1,100)

	ax.set_xscale('log')
	ax.set_xlabel('distance [kpc]')


	p.finalize(fig,type+'cd_relaxed_1e5',save=1)

def plot_tdyn():
	fig,ax = p.makefig(1)
	r_range = np.array([1,2,3,4,5])
	vc_range = np.linspace(30,200,100)

	for r in r_range:
		tdyn_range = m.dynamical_time(r,vc_range) / (1.e6)
		ax.plot(vc_range,tdyn_range,label='r = '+str(r)+' kpc')

	# ax.set_xscale('log')
	ax.legend(frameon=False,prop={'size':11})
	ax.set_xlabel(r'$v_\mathregular{circ}$ [km s$^{-1}$]')
	
	ax.set_yscale('log')
	ax.set_ylim(1e1,5e4)
	ax.set_ylabel(r'$t_\mathregular{dyn}$')
	ax.set_yticks([1e1,1e2,1e3,1e4])
	ax.set_yticklabels(['10 Myr','100 Myr','1 Gyr'])

	p.finalize(fig,'tdyn',save=0)
			
def fit_NFW(sim,snapnum):
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	if not(sim in models):
		raise ValueError('please use a sim in the list of models')

	i = np.where(models==sim)[0][0]
	G = m.Gprime	# kpc * (km/s)^2 / Msun
	H = 70./1000. 	# 70 km/s/Mpc / 1000 kpc/Mpc = 0.07 km/s/kpc
	rho_crit = 3*H**2 / (8*np.pi*G) 

	f = h5py.File(d.datdir+'massprofiles_'+savenames[i]+'.hdf5','r')
	print(d.datdir+'massprofiles_'+savenames[i]+'.hdf5')
	drange = np.array(f['drange'])
	dark_profile = np.array(f['dark'])[snapnum]
	f.close()

	vols = 4./3.*np.pi*(drange**3)
	rho_DM = dark_profile/vols
	ax.plot(drange,rho_DM,label=r'$\rho_\mathregular{DM}$')

	r200 = drange[(rho_DM >= 200*rho_crit)][-1]
	drange_fit = drange[(drange > 1)]
	rho_DM_fit = rho_DM[(drange > 1)]
	print(r200)
	
	# c_arr = np.linspace(1,30,100)
	# c_arr = np.arange(1,300)
	# difference = np.array([])	
	# for c in c_arr:
	# 	attempt = nfw.get_rho_NFW(c,r200,h=0.7,r_array=drange_fit)
	# 	difference = np.append(difference,np.abs(np.sum(attempt - rho_DM_fit)))
	# c_best = c_arr[np.argmin(difference)]

	c = 10
	rho_0_range = np.logspace(3,10,200)

	difference = np.array([])
	for rho_0 in rho_0_range:
		attempt = nfw.get_rho_NFW(c,r200,rho_0,h=0.7,r_array=drange_fit)
		difference = np.append(difference,np.abs(np.sum(attempt - rho_DM_fit)))

	best_rho_0 = rho_0_range[np.argmin(difference)]

	#-----------------------------------------------------
	rho_0_range = np.logspace(np.log10(best_rho_0/10),np.log10(best_rho_0*10),200)

	difference = np.array([])
	for rho_0 in rho_0_range:
		attempt = nfw.get_rho_NFW(c,r200,rho_0,h=0.7,r_array=drange_fit)
		difference = np.append(difference,np.abs(np.sum(attempt - rho_DM_fit)))

	best_rho_0 = rho_0_range[np.argmin(difference)]

	rho_NFW_plt = nfw.get_rho_NFW(c,r200,best_rho_0,h=0.7,r_array=drange)	
	ax.plot(drange,rho_NFW_plt,label=r'$\rho_\mathregular{NFW}$')


	ax.annotate('c = '+str(c),xy=(0.05,0.15),xycoords='axes fraction',fontsize=11,color='k')
	ax.annotate(r'$\rho_0$ = '+m.scinote(best_rho_0),xy=(0.05,0.1),xycoords='axes fraction',fontsize=11,color='k')


	ax.legend(prop={'size':11})

	ax.set_xlim(0.1,100)
	ax.set_xscale('log')
	# ax.set_xlabel('$r$ [kpc]')
	ax.set_xlabel('r [kpc]')
	ax.set_xticks([0.1,1,10])
	ax.set_xticklabels([0.1,1,10])

	ax.set_ylim(1e6,1e9)
	ax.set_yscale('log')
	ax.set_ylabel(r'$\rho$ [M$_\odot$ kpc$^{-3}$]')

	p.finalize(fig,'fit_NFW',save=0)















#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# fit_NFW('fiducial_1e6',2)

# print(m.scinote(nfw.M_NFW_in_r(15,55,55)))

# for n in range(119,401):
# 	alpha_proj('vareff_v2_1e5',n)



panel_projection_single('')








#--------------------------------------------------------------------------------------------------
print('\n')
pdb.set_trace()
