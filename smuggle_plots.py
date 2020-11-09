from __future__ import print_function
import os, sys, pdb, h5py, warnings, time, math, snapHDF5, inspect
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
import fit
from os import listdir
from os.path import isfile, join
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
h=0.7
warnings.filterwarnings("ignore")

from datetime import datetime
hostname = os.uname()[1]
monthnum = int(str(datetime.now()).split(' ')[0].split('-')[1]) - 1
monthlist = np.array(['01.jan','02.feb','03.mar','04.apr','05.may','06.jun','07.jul','08.aug','09.sep','10.oct','11.nov','12.dec'])
month = monthlist[monthnum]
rows, columns = os.popen('stty size', 'r').read().split()
rows = np.int(rows)
columns = np.int(columns)

green = '#8bde8b'
yellow = '#ffde8b'
orange = '#ffa58b'
red = '#ff6a8b'
blue = '#94d0ff'
purple = '#966bff'

# plt.clf()
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# colors_list =  np.array(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
# 	'#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', 
# 	'#9a6324', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#000000'])

# whichsims = 'all'
# whichsims = 'fiducial_1e6'
# whichsims = 'fixMorph_1e6'
# whichsims = 'ff_eSF_1e6'
# whichsims = 'ff_eSF_var_1e5'
# whichsims = 'ff_ill_1e6'
# whichsims = 'f_eSF_var_eSF2_1e6'

# whichsims = 'ff_eSF_var_1e6'
# whichsims = 'eSF_eSFramp_eSF2_1e6'
# whichsims = 'ff_eSF_eSF2_1e6'
# whichsims = 'ff_ill_rho_1e6'
# whichsims = 'mainones_1e6'
# whichsims = 'newsample_1e6'
# whichsims = 'eSF2vareffrho_1e6'
whichsims = 'mainsmuggle_1e6'

# whichsims = 'eSF_runs_1e5'

save = False
# save = True

save_pdf = False
# save_pdf = True


#savenames should be replaced with models
#						models			   #models_label                            #colors_list       alpha    lw
#						directory				
simstuff =  np.array([['fiducial_1e6', 	   'fiducial',                               'black',          '1.0',    '2.0'],
				   	  ['illustris_1e6',    'illustris',                              'limegreen',      '1.0',    '1.6'],
				   	  ['rho0.1_1e6', 	   'rho0.1',                                 'slateblue',      '1.0',    '1.6'],
				   	  ['rho0500_1e6', 	   'rho500',                                 'darkviolet',     '1.0',    '1.6'],
				   	  ['rho1000_1e6', 	   'rho1000',                                'deeppink',       '1.0',    '1.6'],
				   	  ['eSF100_1e6',	   'eSF100',                                 'red',            '1.0',    '1.6'],
				   	  ['eSF100_1e6_2',	   'eSF100 v2',                              'orange',         '1.0',    '1.6'],
					  ['vareff_1e6', 	   'vareff',                                 'deepskyblue',    '1.0',    '1.6'],
				   	  ['fiducial_1e6_fix', 'fiducial 1e6, MeanVol fix',              'dodgerblue',     '1.0',    '1.6'],
				   	  ['eSFramp_1e6',      'eSFramp t0.1 1e6',                       'green',          '1.0',    '1.6'],
				   	  ['eSFramp03_1e6',    'eSFramp t0.3 1e6',                       'gold',           '1.0',    '1.6'],
				   	  ['dwarf_rlx_1e6',    'adiabatic',                              'teal',           '0.6',    '1.3'],
				   	  ['tiny_1e6',         'tiny 1e6',                               'orange',         '1.0',    '1.6'],
				   	  

					  ['fiducial_1e5',	   'fiducial 1e5',                           'black',          '1.0',    '1.6'],
					  ['illustris_1e5',    'illustris 1e5',                          'limegreen',      '1.0',    '1.6'],
					  ['rho0.1_1e5', 	   'rho0.1 1e5',                              purple,          '1.0',    '1.6'],
					  ['rho10_1e5',        'rho10 1e5',                              'violet',         '1.0',    '1.6'],
					  ['rho0500_1e5',      'rho500 1e5',                             'darkviolet',     '1.0',    '1.6'],
					  ['rho1000_1e5',      'rho1000 1e5',                            'magenta',        '1.0',    '1.6'],
					  ['eSF100_1e5', 	   'eSF100 1e5',                             'red',            '1.0',    '1.6'],					  #c9304f
					  ['eSFramp_1e5',      'eSFramp t0.1 1e5',                       'orange',         '1.0',    '1.6'],
					  ['eSFramp03_1e5',    'eSFramp t0.3 1e5',                       'gold',           '1.0',    '1.6'],
					  ['eSFstep_1e5',      'eSFstep t0.1 1e5',                       'greenyellow',    '1.0',    '1.6'],
					  ['eSF100_e0t1_1e5',  'eSF100, no SF t>1',                      'lawngreen',      '1.0',    '1.6'],
					  ['vareff_1e5', 	   'var. eff. 1e5',                          'deepskyblue',    '1.0',    '1.6'],
					  ['fiducial_1e5_fix', 'MeanVol fix, fiducial 1e5',              'dodgerblue',     '1.0',    '1.6'],
					  ['tiny_1e5',         'tiny 1e5',                               'orange',         '1.0',    '1.6'],
					  

				   	  ['cd_rlx_1e5', 	   'compact adiabatic 1e5',                  'teal',           '1.0',    '1.2'],
				   	  ['cd_rlx_sf_1e5',    'compact relaxed 1e5',                    'blue',           '1.0',    '1.2'],
				   	  ['rlx_1e5',  		   'adiabatic 1e5',                          'orange',         '1.0',    '1.2'],
				   	  ['rlx_sf_1e5', 	   'relaxed 1e5',                            'red',            '1.0',    '1.2']
				])

models = simstuff[:,0]
models_label = simstuff[:,1]
colors_list = simstuff[:,2]
alphas_list = simstuff[:,3].astype(np.float)
lw_list = simstuff[:,4].astype(np.float)

# if whichsims=='fixMorph_1e6' or whichsims=='newsample_1e6':
# 	sel = np.in1d(models,np.array(['fiducial_1e6','illustris_1e6']))
# 	alphas_list[sel] = 0.2
# 	lw_list[sel] = 3


if whichsims in models:					selsims = np.array([whichsims])
elif whichsims=='all':					selsims = models
elif whichsims=='fixMorph_1e5':			selsims = np.array(['fiducial_1e5','illustris_1e5','rho0.1_1e5','eSF100_1e5','vareff_1e5'])
elif whichsims=='fixISM_1e5':			selsims = np.array(['fiducial_1e5','compact_1e5','tiny_1e5'])
elif whichsims=='fixMorph_1e6':			selsims = np.array(['fiducial_1e6','illustris_1e6', 'rho0.1_1e6', 'eSF100_1e6', 'vareff_1e6'])
elif whichsims=='fixISM_1e6':			selsims = np.array(['fiducial_1e6','compact_1e6','tiny_1e6'])
elif whichsims=='ff_converge':			selsims = np.array(['fiducial_1e6','fiducial_1e5'])
elif whichsims=='compare_rlx_1e5':		selsims = np.array(['fiducial_1e5','cd_rlx_sf_1e5','rlx_sf_1e5'])
elif whichsims=='ff_ill_1e6':			selsims = np.array(['fiducial_1e6','illustris_1e6'])
elif whichsims=='compare_eSFramp_1e5':  selsims = np.array(['fiducial_1e5','eSF100_1e5','vareff_1e5','eSFramp_1e5','eSFramp03_1e5'])
elif whichsims=='compare_new_1e5':		selsims = np.array(['fiducial_1e5','rho0500_1e5','rho1000_1e5','eSFramp_1e5','eSFramp03_1e5'])
elif whichsims=='compare_rho_1e5':		selsims = np.array(['fiducial_1e5','illustris_1e5','rho0.1_1e5','rho0500_1e5','rho1000_1e5'])
elif whichsims=='compare_rho_1e6':		selsims = np.array(['fiducial_1e6','illustris_1e6','rho0.1_1e6','rho0500_1e6','rho1000_1e6'])
elif whichsims=='compare_esf_1e5':		selsims = np.array(['fiducial_1e5','eSF100_1e5','eSFramp_1e5','eSFramp03_1e5'])
elif whichsims=='compare_esf_1e6':		selsims = np.array(['fiducial_1e6','eSF100_1e6','eSF100_1e6_2','eSFramp_1e6','eSFramp03_1e6'])
elif whichsims=='compare_fix_1e6':	    selsims = np.array(['fiducial_1e6','fiducial_1e6_fix'])
elif whichsims=='compare_rlx_1e6':	    selsims = np.array(['fiducial_1e6','illustris_1e6','dwarf_rlx_1e6'])
elif whichsims=='compare_tiny_1e6':	    selsims = np.array(['fiducial_1e6','tiny_1e6'])
elif whichsims=='compare_tiny_1e5':	    selsims = np.array(['fiducial_1e5','tiny_1e5'])
elif whichsims=='all_1e6':		    	selsims = np.array(['fiducial_1e6','illustris_1e6', 'rho0.1_1e6', 'eSF100_1e6', 'vareff_1e6','tiny_1e6'])
elif whichsims=='all_1e5':		    	selsims = np.array(['fiducial_1e5','illustris_1e5', 'rho0.1_1e5', 'eSF100_1e5', 'vareff_1e5','tiny_1e5'])

elif whichsims=='smuggle_1e6':	    	selsims = np.array(['fiducial_1e6','rho0.1_1e6', 'eSF100_1e6', 'vareff_1e6','tiny_1e6'])
elif whichsims=='smuggle_1e5':	    	selsims = np.array(['fiducial_1e5','rho0.1_1e5', 'eSF100_1e5', 'vareff_1e5','tiny_1e5'])

elif whichsims=='ff+tiny_1e6':	    	selsims = np.array(['fiducial_1e6','tiny_1e6'])
elif whichsims=='ff+tiny_1e5':	    	selsims = np.array(['fiducial_1e5','tiny_1e5'])

elif whichsims=='ff_eSF_1e6':	    	selsims = np.array(['fiducial_1e6','eSF100_1e6'])
elif whichsims=='ff_eSF_var_1e6':	   	selsims = np.array(['fiducial_1e6','eSF100_1e6','vareff_1e6'])
elif whichsims=='ff_ill_rho_1e6':	   	selsims = np.array(['fiducial_1e6','illustris_1e6','rho0.1_1e6'])
elif whichsims=='ff_eSF_var_eSF2_1e6':	selsims = np.array(['fiducial_1e6','eSF100_1e6','vareff_1e6','eSF100_1e6_2'])
elif whichsims=='ff_eSF_eSF2_1e6':		selsims = np.array(['fiducial_1e6','eSF100_1e6','eSF100_1e6_2'])
elif whichsims=='eSF_eSFramp_eSF2_1e6': selsims = np.array(['eSF100_1e6','eSFramp03_1e6','eSF100_1e6_2'])
elif whichsims=='mainones_1e6': 		selsims = np.array(['fiducial_1e6','illustris_1e6','rho0.1_1e6','vareff_1e6','eSF100_1e6','eSF100_1e6_2','eSFramp03_1e6'])
elif whichsims=='newsample_1e6':		selsims = np.array(['fiducial_1e6','illustris_1e6', 'rho0.1_1e6', 'eSF100_1e6_2', 'vareff_1e6'])
elif whichsims=='eSF2vareffrho_1e6':	selsims = np.array(['rho0.1_1e6', 'eSF100_1e6_2', 'vareff_1e6'])
elif whichsims=='mainsmuggle_1e6':		selsims = np.array(['fiducial_1e6', 'rho0.1_1e6', 'eSF100_1e6','eSF100_1e6_2', 'vareff_1e6'])

elif whichsims=='eSF_runs_1e5':		    selsims = np.array(['fiducial_1e5','eSF100_1e5','eSF100_e0t1_1e5','vareff_1e5'])


elif whichsims=='ff_eSF_var_1e5':	   	selsims = np.array(['fiducial_1e5','eSF100_1e5','vareff_1e5'])


else: raise ValueError('unknown whichsims')


mask = np.in1d(models,selsims)

if np.count_nonzero(mask) < len(selsims): 
	notinstr = m.tostring(selsims[np.in1d(selsims,models,invert=True)])
	raise ValueError('whichsims contains the following unknown simulations: '+notinstr)

models = models[mask]
models_label = models_label[mask]
colors_list = colors_list[mask]
alphas_list = alphas_list[mask]
lw_list = lw_list[mask]
print('whichsims = '+whichsims)
# print(models)

#--------------------------------------------------------------------------------------------------
#---plotting-functions-----------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
# text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor=colors_list[i], edgecolor='white'), path_effects.Normal()])

def sfr_proj(sim,snapnum,do_grid=0,do_hist=1,bound=5):
	print('plotting sfr_proj')
	if not(sim in models):
		raise ValueError('please choose one of the available simulations')

	i = np.where(models==sim)[0][0]

	fig,ax = p.makefig(1,figx=8,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	plt.style.use('dark_background')

	#create SFR space for colormap to sample
	# c = np.linspace(-6,-5,1000)
	c = np.linspace(-4,-2,1000)
	norm = mplcolors.Normalize(vmin=c.min(), vmax=c.max())
	cmap = cm.ScalarMappable(norm=norm, cmap=cm.plasma_r)
	cmap.set_array([])

	#----------------------------------------------------------------------------------------------
	cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
	CoM_all = np.array(cm_file['CoM'])
	cm_file.close()

	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
	# gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
	gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h - CoM_all[snapnum]
	gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
	sel_bound = (np.abs(gas_pos[:,0]) < bound) & (np.abs(gas_pos[:,1]) < bound) & (np.abs(gas_pos[:,2]) < bound) & (gas_sfr > 0)
	gas_pos = gas_pos[sel_bound]
	gas_sfr = gas_sfr[sel_bound]
	#----------------------------------------------------------------------------------------------
	if do_hist:
		ax.hist(np.log10(gas_sfr),bins=30,histtype='step',color='turquoise')
		ax.annotate(r'$n = $'+str(np.count_nonzero(sel_bound)),xy=(0.7,0.9),xycoords='axes fraction',color='white',size=12)

	elif do_grid:
		xbins = np.linspace(-bound,bound,200)
		ybins = np.linspace(-bound,bound,200)
		bw = (xbins[1] - xbins[0])/2

		total = len(xbins)-1
		
		allvalues = np.array([])

		for j in range(total):
			selx = (gas_pos[:,0] > xbins[j]) & (gas_pos[:,0] < xbins[j+1])
			thisx = xbins[j]+bw

			printthing = str(np.round(j/2,0))+'%'
			sys.stdout.write(printthing)
			sys.stdout.flush()
			sys.stdout.write("\b" * (len(printthing)))

			for k in range(len(ybins)-1):
				sely = (gas_pos[:,1] > ybins[k]) & (gas_pos[:,1] < ybins[k+1])
				thisy = ybins[k]+bw

				this_sel = selx & sely & (gas_pos[:,2] > -1) & (gas_pos[:,2] < 1)

				if (np.count_nonzero(this_sel) > 0):
					meanvalue = np.mean(gas_sfr[this_sel])
					ax.plot(thisx,thisy,'o',ms=2,mew=0,c=cmap.to_rgba(m.find_nearest(c,meanvalue)))
	else:
		maxnum = float(len(gas_pos[:,0]))
		for j in np.arange(maxnum):
			printthing = str(int(j)) +' / ' + str(int(maxnum))
			sys.stdout.write(printthing)
			sys.stdout.flush()
			sys.stdout.write("\b" * (len(printthing)))
			ax.plot(gas_pos[j,0],gas_pos[j,1],'o',ms=1,mew=0,c=cmap.to_rgba(m.find_nearest(c,np.log10(gas_sfr[j]))))

	
	#----------------------------------------------------------------------------------------------
	if do_hist:
		ax.set_xlabel(r'log$_{10}$(SFR) [M$_\odot$ yr$^{-1}$]')
		fname = 'sfr_hist_'+sim+'_'+str(snapnum).zfill(3)
	else:
		fname = 'sfr_proj_'+sim+'_'+str(snapnum).zfill(3)
		fig.colorbar(cmap,label=r'log$_{10}$(SFR) [M$_\odot$ yr$^{-1}$]')

		ax.set_xlim(-bound,bound)
		ax.set_xlabel('x [kpc]')
		ax.set_ylim(-bound,bound)
		ax.set_ylabel('y [kpc]')

	ax.annotate(models_label[i],xy=(0.05,0.95),xycoords='axes fraction',fontsize=11,color='white')
	ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.05,0.9),xycoords='axes fraction',fontsize=11,color='white')

	# p.finalize(fig,fname,save=save)
	fname = str(snapnum).zfill(3)
	sd = '/home/ejahn003/movie_frames/sfr_proj_'+sim+'/'
	print('saving figure: '+fname+'.png')
	plt.savefig(sd+fname+'.png',format='png',dpi=200)

def calculate_rho_hist_timeavg(sim):

	print('calculating rho hist time average for '+sim)

	scut = 10

	rho_bins = np.logspace(-6,6,100)
	rho_bin_means = np.array([])

	for j in range(len(rho_bins)-1):
		rho_mean = (rho_bins[j] + rho_bins[j+1])/2
		rho_bin_means = np.append(rho_bin_means,rho_mean)

	for sim in models:
		i = np.where(models==sim)[0][0]

		rhofile =  h5py.File(d.datdir+'rho_1kpc_'+sim+'.hdf5','r')
		k = np.array(rhofile.keys())
		kl = np.sort(k)
		max_snap = int(kl[-1].split('_')[-1]) + 1

		print('snapshots '+str(scut)+' to '+str(max_snap))

		num_list_all = np.array([])

		for snapnum in np.arange(scut,max_snap):
			rho = np.array(rhofile['gas_rho_'+str(snapnum).zfill(3)])
			num_list = np.array([])

			for j in range(len(rho_bins)-1):
				sel = (rho > rho_bins[j]) & (rho < rho_bins[j+1]) 
				num_list = np.append(num_list,np.count_nonzero(sel))

			if snapnum==scut: 	num_list_all = num_list
			else:				num_list_all = np.vstack((num_list_all,num_list))

		num_means = np.array([])
		for j in range(num_list_all.shape[1]):
			num_means = np.append(num_means,np.mean(num_list_all[:,j]))

def cmtest():
	fig,ax = p.makefig(1)
	models = np.array(['fiducial','illustris','rho0.1','eSF100','vareff'])
	a = np.arange(len(models))

	for i in a:
		c = cm.CMRmap(i/(len(a)-0.5),1)
		ax.plot(a,i*a,color=c,lw=3)
		text = ax.annotate(models[i],xy=(0.05,0.95-(i*0.05)),xycoords='axes fraction',size=11,color=c)
		# text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor=c, 
		# 	edgecolor='black'), path_effects.Normal()])

	p.finalize(fig,'cmtest',save=0)

def compare_dcuts_twopanel_cores(sim):
	fig,axarr = p.makefig('2_vert',height=1,figx=7,figy=8)
	fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	dcut_array = np.array([1,2,3,4,5,6,7,8,9,10])

	xloc = np.array([0.70,0.70,0.70,0.70,0.70,0.80,0.80,0.80,0.80,0.80])
	yloc = np.array([0.25,0.20,0.15,0.10,0.05,0.25,0.20,0.15,0.10,0.05])+0.05

	for i,this_dcut in enumerate(dcut_array):
		this_color = cm.plasma(i/float(len(dcut_array)),1)

		#---plot-core-radius-on-top-panel----------------------------------------------------------
		nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut'+str(this_dcut)+'_hires.hdf5','r')
		time = np.array(nf['time'])
		core_radius = np.array(nf['core_radius'])
		nf.close()
		# mean_rcore = np.mean(core_radius[time > 0.75])
		# axarr[0].axhline(y=mean_rcore,ls=':',lw=1.5,color=colors_list[i],alpha=alphas_list[i])
		axarr[0].plot(time,core_radius,c=this_color,lw=1,alpha=0.7)

		#---plot-power-law-slope-on-bottom-panel----------------------------------------------------
		f = h5py.File(d.datdir+'powerlawslope_nfwdcut'+str(this_dcut)+'_'+sim+'_hires.hdf5')
		slope = np.array(f['slope'])
		time = np.array(f['time'])
		f.close()

		axarr[1].plot(time,slope,c=this_color,lw=1,alpha=0.7)

		axarr[1].annotate(str(this_dcut)+' kpc',xy=(xloc[i],yloc[i]),xycoords='axes fraction',color=this_color,size=11)

	#--set-up-frames-------------------------------------------------------------------------------
	y = np.mean(np.array([0.25,0.20,0.15,0.10,0.05])+0.05)
	axarr[1].annotate('nfw_dcut = ',xy=(0.5,y),xycoords='axes fraction',color='k',size=11)
	axarr[0].annotate(models_label[np.where(models==sim)[0][0]],xy=(0.8,0.9),xycoords='axes fraction',color='k',size=13)
	
	axarr[0].set_ylim(0,0.7)
	# ax.set_yticks(np.arange(0,1.1,0.1),minor=True)
	# ax.set_yticks([0,0.25,0.5,0.75,1])
	axarr[0].set_yticks(np.arange(0,0.7,0.05),minor=True)
	axarr[0].set_yticks(np.arange(0,0.8,0.1))
	axarr[0].set_yticklabels([0,100,200,300,400,500,600,700])
	axarr[0].set_ylabel(r'r$_\mathregular{core}$ [pc]')
	# ax.set_ylim(0,1)
	
	axarr[0].legend(frameon=False,loc='upper left',prop={'size':11})
	
	axarr[1].set_xlim([0,2.86])
	axarr[1].set_xticks([0,0.5,1,1.5,2,2.5])
	axarr[1].set_xticks([0.25,0.75,1.25,1.75,2.25,2.75],minor=True)
	axarr[1].set_xlabel(r'time [Gyr]')

	axarr[1].set_ylabel(r'$\alpha$')
	axarr[1].set_ylim(-0.8,0.1)
	axarr[1].set_yticks([-0.8,-0.6,-0.4,-0.2,0])
	axarr[1].set_yticks([-0.7,-0.5,-0.3,-0.1],minor=True)

	axarr[1].legend(prop={'size':10},loc='lower right',frameon=False)

	fname = 'rcore_slope_compare_dcuts_' + whichsims

	p.finalize(fig,fname,save=save,save_pdf=save_pdf,tight=True)

def core_radius(do_bins=0,hires=0,do_dcut5=0):
	print('plotting core radius')

	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# print(ptypes)
	for i,sim in enumerate(models):
		print(sim)
		if hires: nfname = d.datdir+'coreradius_'+sim+'_dcut3_hires.hdf5'
		else:     nfname = d.datdir+'coreradius_'+sim+'_dcut3.hdf5'
		print(nfname)
		nf = h5py.File(nfname,'r')
		time = np.array(nf['time'])
		core_radius = np.array(nf['core_radius'])
		nfw_dcut = np.array(nf['nfw_dcut']).astype(float)
		ratio_cut = np.array(nf['ratio_cut']).astype(float)
		nf.close()

		#--find-average-rcore-after-0.75Gyr-(roughly-when-it-stabilizes)---------------------------
		mean_rcore = np.mean(core_radius[time > 0.75])
		ax.axhline(y=mean_rcore,ls=':',lw=1.5,color=colors_list[i],alpha=alphas_list[i])

		#---smooth-core-radius-by-binning----------------------------------------------------------
		if do_bins:
			# timebins = np.linspace(0, 2.8, 140)
			timebins = np.arange(0,2.86,0.025)
			binwidth = (timebins[1] - timebins[0])/2

			rcore_mean = np.array([])
			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				sel = (time > leftbin) & (time < rightbin)
				rcore_mean = np.append(rcore_mean, np.mean(core_radius[sel]))

			time_plt = timebins[0:-1]+binwidth
			ax.plot(time_plt,rcore_mean,color=colors_list[i],lw=lw_list[i],alpha=alphas_list[i],ls='-',zorder=10*(i+1))

			#--plot-points-at-selected-snapshots-with-binned-values--------------------------------
			if (sim=='fiducial_1e6') and whichsims=='ff_ill_1e6':
				points = np.array([1,65,169,400])
				sel = np.in1d(np.arange(401),points)
				
				pointtimes = time[sel]
				newtimeplt = np.array([])
				new_rcores = np.array([])
				
				for t in pointtimes:
					newtimeplt = np.append(newtimeplt,m.find_nearest(time_plt,t))
					new_rcores = np.append(new_rcores,rcore_mean[m.find_nearest(time_plt,t,getindex=1)])

				ax.plot(newtimeplt,new_rcores,'s',mew=0.5,ms=10,mec='white',mfc='k',zorder=10*(i+1))

		#---plot-core-radius-without-smoothing-----------------------------------------------------
		else:
			ax.plot(time,core_radius,colors_list[i],lw=lw_list[i],alpha=alphas_list[i],zorder=10*(i+1))
			
			#--plot-points-at-selected-snapshots---------------------------------------------------
			if (sim=='fiducial_1e6') and whichsims=='ff_ill_1e6':
				points = np.array([1,65,169,400])
				sel = np.in1d(np.arange(401),points)
				ax.plot(time[sel],core_radius[sel],'s',mew=0.5,ms=10,mec='white',mfc='k',zorder=10*(i+1))

		text = ax.annotate(models_label[i],xy=(0.05,0.95-(0.04*i)),xycoords='axes fraction',fontsize=11,color=colors_list[i],alpha=alphas_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor=colors_list[i], 
			edgecolor='white'), path_effects.Normal()])

	ax.set_ylim(0,0.6)
	ax.set_yticks(np.arange(0,0.6,0.05),minor=True)
	ax.set_yticks(np.arange(0,0.7,0.1))
	ax.set_yticklabels([0,100,200,300,400,500,600])
	ax.set_ylabel(r'r$_\mathregular{core}$ [pc]')
	
	ax.legend(frameon=False,loc='upper left',prop={'size':11})
	ax.set_xlim([0,2.8])
	ax.set_xticks([0,0.5,1,1.5,2,2.5])
	ax.set_xticks([0.25,0.75,1.25,1.75,2.25,2.75],minor=True)
	ax.set_xlabel(r'time [Gyr]')

	if hires: fname = 'radius_coreNFW_' + whichsims + '_hires'
	else:     fname = 'radius_coreNFW_' + whichsims

	p.finalize(fig,fname,save=save,save_pdf=save_pdf,tight=True)

def compare_core_radius(do_bins=True,hires=1):
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# print(ptypes)
	for i,sim in enumerate(models):
		print(sim)
		text = ax.annotate(models_label[i],xy=(0.05,0.95-(0.04*i)),xycoords='axes fraction',fontsize=11,color=colors_list[i],alpha=alphas_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor=colors_list[i], 
			edgecolor='white'), path_effects.Normal()])
		if hires: nfname = d.datdir+'coreradius_'+sim+'_dcut3_hires.hdf5'
		else:     nfname = d.datdir+'coreradius_'+sim+'_dcut3.hdf5'
		print(nfname)
		nf = h5py.File(nfname,'r')
		time = np.array(nf['time'])
		core_radius = np.array(nf['core_radius'])
		nf.close()

		if hires: nfname = d.datdir+'PITS_params_'+sim+'_dcut10_hires.hdf5'
		else:     nfname = d.datdir+'PITS_params_'+sim+'_dcut10.hdf5'
		print(nfname)
		nf = h5py.File(nfname,'r')
		rc_pits = np.array(nf['rc'])
		nf.close()

		#--find-average-rcore-after-0.75Gyr-(roughly-when-it-stabilizes)---------------------------
		# mean_rcore = np.mean(core_radius[time > 0.75])
		# ax.axhline(y=mean_rcore,ls=':',lw=1.5,color=colors_list[i],alpha=alphas_list[i])

		mean_rcpits = np.mean(rc_pits[time > 0.75])
		ax.axhline(y=mean_rcpits,ls=':',lw=1.5,color=colors_list[i],alpha=alphas_list[i])

		#---smooth-core-radius-by-binning----------------------------------------------------------
		if do_bins:
			dt = 0.05
			timebins = np.arange(0,np.amax(time)+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2

			rcore_mean = np.array([])
			rcpits_mean = np.array([])
			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				sel = (time > leftbin) & (time < rightbin)
				rcore_mean = np.append(rcore_mean, np.mean(core_radius[sel]))
				rcpits_mean = np.append(rcpits_mean, np.mean(rc_pits[sel]))

			time_plt = timebins[0:-1]+binwidth
			# ax.plot(time_plt,rcore_mean,color=colors_list[i],lw=lw_list[i],alpha=alphas_list[i],ls='-',zorder=10*(i+1))
			ax.plot(time_plt,rcpits_mean,color=colors_list[i],lw=lw_list[i],alpha=alphas_list[i],ls='-',zorder=10*(i+1))

		#---plot-core-radius-without-smoothing-----------------------------------------------------
		else:
			# ax.plot(time,core_radius,colors_list[i],lw=lw_list[i],alpha=alphas_list[i],zorder=10*(i+1))
			ax.plot(time,rc_pits,colors_list[i],lw=lw_list[i],alpha=alphas_list[i],zorder=10*(i+1))

	# ax.set_ylim(0,0.6)
	# ax.set_yticks(np.arange(0,0.6,0.05),minor=True)
	# ax.set_yticks(np.arange(0,0.7,0.1))
	# ax.set_yticklabels([0,100,200,300,400,500,600])
	ax.set_ylim(0,1.8)
	ax.set_yticks(np.arange(0,1.8,0.1),minor=True)
	ax.set_yticks(np.arange(0,1.8,0.5))
	# ax.set_yticklabels([0,100,200,300,400,500,600])
	ax.set_ylabel(r'r$_\mathregular{core}$ [kpc]')
	
	ax.legend(frameon=False,loc='upper left',prop={'size':11})
	ax.set_xlim([0,2.8])
	ax.set_xticks([0,0.5,1,1.5,2,2.5])
	ax.set_xticks([0.25,0.75,1.25,1.75,2.25,2.75],minor=True)
	ax.set_xlabel(r'time [Gyr]')

	# if hires: fname = 'rcore_PITS_' + whichsims + '_hires'
	fname = 'rcore_PITS_' + whichsims

	p.finalize(fig,fname,save=save,save_pdf=save_pdf,tight=True)

def countparts(snapnum):
	for sim in models:
		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)	

		darkpos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h

		x_cm = np.sum(darkpos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(darkpos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(darkpos[:,2] * darkmass) / np.sum(darkmass)
		cm = np.array([x_cm,y_cm,z_cm]).T

		darkpos = darkpos - cm

		d_dark = np.linalg.norm(darkpos,axis=1)

		print('\n--------------------------')
		print(sim)
		for dist in np.array([0.1,0.3]):
			sel = (d_dark < dist)
			num = np.count_nonzero(sel)
			print('N_dm < '+str(dist)+' kpc: '+str(num))

def density_profile_single(sim,snapnum,doPITS=True):
	print('plotting density profile for '+sim+', snapshot '+str(snapnum).zfill(3))

	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	sn = models_label[np.where(models==sim)[0][0]] + ' snapshot '+str(snapnum).zfill(3)

	text = ax.annotate(sn,xy=(0.05,0.9),xycoords='axes fraction',fontsize=12,color='black')#,alpha=alphas_list[i])
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='black', edgecolor='white'), path_effects.Normal()])

	try:
		f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
	except:
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')

	drange = np.array(f['drange'])
	vols = 4./3.*np.pi*(drange**3)

	gas_profile = np.array(f['gas'])[snapnum]/vols
	dark_profile = np.array(f['dark'])[snapnum]/vols
	type2_profile = np.array(f['type2'])[snapnum]/vols
	type3_profile = np.array(f['type3'])[snapnum]/vols
	type4_profile = np.array(f['type4'])[snapnum]/vols
	f.close()

	profiles = np.vstack((gas_profile, dark_profile, type2_profile, type3_profile, type4_profile))
	colors = np.array(['dodgerblue','black','blueviolet','limegreen','orange'])
	name = np.array(['gas','dm','type2','type3','type4'])

	mask = np.array([False,True,False,False,False])
	profiles = profiles[mask]
	colors = colors[mask]
	name = name[mask]

	for i, profile in enumerate(profiles):
		text = ax.annotate(name[i],xy=(0.8,0.9-(0.06*i)),xycoords='axes fraction',fontsize=12,color=colors[i])#,alpha=alphas_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=colors[i], edgecolor='white'), path_effects.Normal()])
		
		ax.plot(drange,profiles[i],lw=1.6,color=colors[i],zorder=(i+1)*10)

	if doPITS and ('dm' in name):
		text = ax.annotate('PITS',xy=(0.8,0.9-(0.06*(i+1))),xycoords='axes fraction',fontsize=12,color='red')#,alpha=alphas_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='red', edgecolor='white'), path_effects.Normal()])

		drange_fit = drange[(drange < 10)]
		rho_dm_fit = dark_profile[(drange < 10)]
		fitted_pars = fit.pits_sqfit(xdata=drange_fit,ydata=rho_dm_fit,rho0_guess=1e7,rc_guess=1)
		rho_PITS = fit.pits_model(fitted_pars,drange)
		ax.plot(drange,rho_PITS,lw=1.6,color='red',zorder=0)
		ax.axvline(x=fitted_pars[1],ls='--',lw=1,c='red')
		ax.annotate(r'$r_\mathregular{c}$ = '+str(np.round(fitted_pars[1]))+' kpc',xy=(0.1,0.1),xycoords='axes fraction',fontsize=12,color='black')#,alpha=alphas_list[i])
		

	ax.set_xscale('log')
	ax.set_xlim(0.1,10)
	ax.set_xticks([0.1,1,10])
	ax.set_xticklabels(['0.1','1','10'])
	ax.set_xlabel('dist [kpc]')
	
	ax.set_yscale('log')
	ax.set_ylim(1e6,1e10)
	ax.set_ylabel(r'$\rho$ [M$_\odot$ kpc$^{-3}$]')

	p.finalize(fig,'density_profile_'+sim+'_'+str(snapnum).zfill(3),save=save)

def dynamicaltime(snapnum):
	print('plotting dynamicaltime')
	fig,ax = p.makefig(1,figx=5,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for i,sim in enumerate(models):
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		gas_profile = np.array(f['gas'])[snapnum]
		dark_profile = np.array(f['dark'])[snapnum]
		type2_profile = np.array(f['type2'])[snapnum]
		type3_profile = np.array(f['type3'])[snapnum]
		type4_profile = np.array(f['type4'])[snapnum]	
		f.close()

		total_profile = gas_profile + dark_profile + type2_profile + type3_profile + type4_profile
		vols = 4./3.*np.pi*(drange**3)
		total_rho = total_profile / vols
		dark_rho = dark_profile / vols

		# tdyn = np.sqrt(3*np.pi / (16*m.Gsim*total_rho))
		tdyn = np.sqrt(3*np.pi / (16*m.Gsim*dark_rho))

		sel = (drange > 0.1) & (drange < 1)
		print('mean tdyn in inner regions: '+str(np.mean(tdyn[sel])))

		ax.plot(drange,tdyn,c=colors_list[i],lw=lw_list[i],alpha=alphas_list[i])
		ax.annotate(models_label[i],xy=(0.05,0.95-(0.04*i)),xycoords='axes fraction',fontsize=11,color=colors_list[i],alpha=alphas_list[i])


	ax.set_xscale('log')
	ax.set_xlim(0.1,10)
	ax.set_xticks([0.1,1,10])
	ax.set_xticklabels([0.1,1,10])
	ax.set_xlabel(r'$r$ [kpc]')

	ax.set_yscale('log')
	ax.set_ylabel(r'$t_\mathregular{dyn}$ [sec]')

	p.finalize(fig,'dynamicaltime_'+whichsims,save=save)

def four_rho_plots_ratio_ffill(do_slope=True,doPITS=True):
	if not(whichsims=='ff_ill_1e6'):
		raise ValueError('please choose whichsims = ff_ill_1e6. it is currently '+whichsims)

	print('plotting four_rho_plots_ratio_ffill')
	fig, axarr = p.makefig(n_panels='4_horiz_with_ratio',figx=20,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	#---plot-density-profiles----------------------------------------------------------------------
	imax = 169 #rcore = 0.84
	inb = 65   #rcore = 0.41
	imin = 1   #rcore = 0.14
	i_ar = np.array([imin,inb,imax,400]) #array of snapshots to plot

	for i,sim in enumerate(models):
		#---read-all-core-radius-----------------------------------------------------------------------
		nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3.hdf5','r')
		time = np.array(nf['time'])
		core_radius = np.array(nf['core_radius'])
		nfw_dcut = np.array(nf['nfw_dcut']).astype(float)
		nf.close()

		pf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5','r')
		rcore_PITS = np.array(pf['rc'])
		rho0_PITS = np.array(pf['rho0'])
		pf.close()

		pf = h5py.File(d.datdir+'power_radius_'+sim+'.hdf5','r')
		power_radius = np.array(pf['power_radius'])
		pf.close()

		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		dark_mass_all = np.array(f['dark'])
		f.close()

		max_snap = dark_mass_all.shape[0]
		vols = 4./3.*np.pi*(drange**3)

		dark_rho_0 = dark_mass_all[0]/vols
		dark_rho_400 = dark_mass_all[400]/vols
		dark_rho_max = dark_mass_all[imax]/vols
		dark_rho_inb = dark_mass_all[inb]/vols
		dark_rho_min = dark_mass_all[imin]/vols

		dark_rho_ar = np.vstack((dark_rho_min,dark_rho_inb,dark_rho_max,dark_rho_400))

		for j in np.arange(4):
			if j==1: 
				ltt = r'current $\rho_\mathregular{DM}$'
				lt0 = r'initial $\rho_\mathregular{DM}$'
				ltN = 'NFW fit'
				ltR = 'NFW/data'
			else: ltt=lt0=ltN=ltR=''

			rho_DM = dark_rho_ar[j]

			#--fit-NFW---------------------------------------------------------------------------------
			H = 70./1000.
			rho_crit = 3*H**2 / (8*np.pi*m.Gprime) 
			r200 = drange[(rho_DM >= 200*rho_crit)][-1]
			rho_DM_fit = rho_DM[(drange > nfw_dcut)]
			drange_fit = drange[(drange > nfw_dcut)]

			fitted_pars = fit.NFW_sqfit(drange_fit,rho_DM_fit,c_guess=10,rho0_guess=1e7,r200=r200)
			rho_nfw = fit.NFW_model(fitted_pars,drange,r200)

			#--plot-the-profiles-----------------------------------------------------------------------
			if i==0:
				axarr[0,j].plot(drange,dark_rho_0,'-',lw=1,c='grey',label=lt0)
				axarr[0,j].plot(drange,rho_nfw,ls='--',c='#1279ff',label=ltN,lw=1.6)
				axarr[1,j].axhline(y=2,color='grey',ls=':',lw=0.9)

			axarr[0,j].plot(drange,rho_DM,lw=4/(i+1),c=colors_list[i],label=models_label[i],zorder=1000/(i+1))
			axarr[1,j].plot(drange,rho_nfw/rho_DM,lw=2.5,c=colors_list[i],label=ltR)
			
			#---plot-the-core-radius---------------------------------------------------------------
			if doPITS:
				if i==0:
					axarr[0,j].axvline(x=0.556942176925,ls='-.',lw=1,color='grey')
					axarr[1,j].axvline(x=0.556942176925,ls='-.',lw=1,color='grey')

				axarr[0,j].axvline(x=rcore_PITS[i_ar[j]],ls='--',lw=1.5,color=colors_list[i],alpha=0.5)
				axarr[1,j].axvline(x=rcore_PITS[i_ar[j]],ls='--',lw=1.5,color=colors_list[i],alpha=0.5)

				print(sim,i_ar[j])
				print(rcore_PITS[i_ar[j]])

				if rcore_PITS[i_ar[j]] > 0.1:
					axarr[1,j].annotate(r'$r_\mathregular{core}$ = '+str(int(np.round(rcore_PITS[i_ar[j]]*1000,0)))+' pc',
						xy=(rcore_PITS[i_ar[j]]*1.05,3-(2.3*i)),fontsize=11,color=colors_list[i])
				elif rcore_PITS[i_ar[j]] > 0: 
					axarr[1,j].annotate(r'$r_\mathregular{core}$ = '+str(int(np.round(rcore_PITS[i_ar[j]]*1000,0)))+' pc',
						xy=(0.11,3-(2.3*i)),fontsize=11,color=colors_list[i])
				elif rcore_PITS[i_ar[j]] < 0: 
					axarr[1,j].annotate(r'$r_\mathregular{core}$ = 0.0 pc', xy=(0.11,3-(2.3*i)),fontsize=11,color=colors_list[i])

			#---plot-the-slope---------------------------------------------------------------------
			if do_slope and (j>=1) and not(doPITS):
				sel = (drange < core_radius[i_ar[j]])
				drange_fit = drange[sel]
				rho_dm_fit = rho_DM[sel]

				fitted_pars = fit.powerlaw_sqfit(xdata=drange_fit,ydata=rho_dm_fit,a_guess=1e8,k_guess=-0.1)
				points = np.logspace(np.log10(0.11),np.log10(core_radius[i_ar[j]]),2)
				plp = fit.powerlaw_model(fitted_pars,points)
				axarr[0,j].plot(points,plp,'-s',lw=1.5,mew=0,ms=5,c='gold')
				axarr[0,j].annotate(r'$\alpha$ = '+str(np.round(fitted_pars[1],2)),xy=(0.5,0.16-(i*0.05)),xycoords='axes fraction',fontsize=12,color=colors_list[i],alpha=1)

			elif do_slope and doPITS:
				sel = (drange > power_radius[i_ar[j]]) & (drange < 0.556942176925)
				drange_fit = drange[sel]
				rho_dm_fit = rho_DM[sel]

				fitted_pars = fit.powerlaw_sqfit(xdata=drange_fit,ydata=rho_dm_fit,a_guess=1e8,k_guess=-0.1)
				points = np.logspace(np.log10(0.11),np.log10(0.556942176925),2)
				plp = fit.powerlaw_model(fitted_pars,points)
				axarr[0,j].plot(points,plp,'-s',lw=1.5,mew=0,ms=5,c='gold',zorder=1e4)
				axarr[0,j].annotate(r'$\alpha$ = '+str(np.round(fitted_pars[1],2)),xy=(0.5,0.16-(i*0.05)),xycoords='axes fraction',fontsize=12,color=colors_list[i],alpha=1)

			#--plot-PITS-----------------------------------------------------------------------------
			if doPITS and i==0:
				fitted_pars = np.array([rho0_PITS[i_ar[j]],rcore_PITS[i_ar[j]]])
				rho_PITS = fit.pits_model(fitted_pars,drange)
				axarr[0,j].plot(drange,rho_PITS,lw=1.6,color='red',zorder=0,label='PITS')

			#--adjust-plot-----------------------------------------------------------------------------
			if i==0:
				# p.clear_axes(axarr[j])
				axarr[0,j].set_xscale('log')
				axarr[0,j].set_xlim(1e-1,1e1)
				snapfile = d.smuggledir + sim + '/snapshot_'+str(i_ar[j]).zfill(3)
				header = snapHDF5.snapshot_header(snapfile)
				time = header.time/0.7
				text = axarr[0,j].annotate(r'$t=$'+str(np.round(time,2))+' Gyr',xy=(0.05,0.92),xycoords='axes fraction',fontsize=11,color='black')
				text.set_path_effects([path_effects.PathPatchEffect(linewidth=5, facecolor='black', 
					edgecolor='white'), path_effects.Normal()])

	#---finalize-plot------------------------------------------------------------------------------
	axarr[0,1].legend(loc='upper right',frameon=False,prop={'size':11})
	axarr[1,0].legend(loc='upper right',frameon=False,prop={'size':11})

	axarr[0,0].set_yscale('log')
	axarr[0,0].set_ylabel(r'$\rho_\mathregular{DM}$ [M$_\odot$ kpc$^{-3}$]')
	axarr[0,0].set_ylim(1e6,3e9)
	axarr[0,0].set_yticks([1e6,1e7,1e8,1e9])

	axarr[1,0].set_ylabel(r'$\rho_\mathregular{NFW}$ / $\rho_\mathregular{DM}$')
	axarr[1,0].set_yscale('log')
	axarr[1,0].set_ylim(0.5,5)
	axarr[1,0].set_yticks([0.5,1,2])
	axarr[1,0].set_yticklabels([0.5,1,2])

	axarr[1,0].set_xlabel('distance [kpc]')

	axarr[0,0].set_xticks([1e-1,1e0]);		axarr[0,0].set_xticklabels(['0.1','1'])
	axarr[0,1].set_xticks([1e-1,1e0]);		axarr[0,1].set_xticklabels(['0.1','1'])
	axarr[0,2].set_xticks([1e-1,1e0]);		axarr[0,2].set_xticklabels(['0.1','1'])
	axarr[0,3].set_xticks([1e-1,1e0,1e1]);	axarr[0,3].set_xticklabels(['0.1','1','10'])
	if doPITS:
		fname = 'four_rho_ratio_slope_PITS_ffill1e6'
	else:
		fname = 'four_rho_ratio_slope_ffill1e6'
	p.finalize(fig,fname,save=save,save_pdf=save_pdf)

def four_rho_plots_ratio_fixMorph(do_slope=True):
	if not(whichsims=='fixMorph_1e6'):
		raise ValueError('please choose whichsims = fixMorph_1e6. it is currently '+whichsims)

	print('plotting four_rho_plots_ratio_fixMorph')
	fig, axarr = p.makefig(n_panels='4_horiz_with_ratio',figx=20,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# sim = 'fiducial_1e6'

	for i,sim in enumerate(models):
		if i<2:	continue #skip fiducial and illustris

		text = axarr[0,1].annotate(models_label[i], xy=(0.65, 0.93-0.05*(i-2)), xycoords='axes fraction', fontsize=12,color=colors_list[i],zorder=1e4)

		print(sim)
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		dark_mass_all = np.array(f['dark'])
		f.close()

		max_snap = dark_mass_all.shape[0]
		vols = 4./3.*np.pi*(drange**3)

		#---read-all-core-radius-----------------------------------------------------------------------
		nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut2.hdf5','r')
		time = np.array(nf['time'])
		core_radius = np.array(nf['core_radius'])
		nfw_dcut = np.array(nf['nfw_dcut']).astype(float)
		nf.close()

		#---plot-density-profiles----------------------------------------------------------------------
		imax = 227
		inb = 169  
		imin = 65

		dark_rho_0 = dark_mass_all[0]/vols
		dark_rho_400 = dark_mass_all[400]/vols
		dark_rho_max = dark_mass_all[imax]/vols
		dark_rho_inb = dark_mass_all[inb]/vols
		dark_rho_min = dark_mass_all[imin]/vols

		dark_rho_ar = np.vstack((dark_rho_min,dark_rho_inb,dark_rho_max,dark_rho_400))
		i_ar = np.array([imin,inb,imax,400]) #array of snapshots to plot

		hl = np.array([3.5,2.5,0.8])

		for j in np.arange(4):
			#--fit-NFW---------------------------------------------------------------------------------
			rho_DM = dark_rho_ar[j]
			H = 70./1000.
			rho_crit = 3*H**2 / (8*np.pi*m.Gprime) 
			r200 = drange[(rho_DM >= 200*rho_crit)][-1]
			rho_DM_fit = rho_DM[(drange > nfw_dcut)]
			drange_fit = drange[(drange > nfw_dcut)]

			fitted_pars = fit.NFW_sqfit(drange_fit,rho_DM_fit,c_guess=10,rho0_guess=1e7,r200=r200)
			rho_nfw = fit.NFW_model(fitted_pars,drange,r200)

			#--plot-the-stuff----------------------------------------------------------------------
			axarr[0,j].plot(drange,rho_DM,lw=3.5,c=colors_list[i])
			axarr[0,j].axvline(x=core_radius[i_ar[j]],ls='-.',lw=1.5,color=colors_list[i])
			axarr[0,j].plot(drange,rho_nfw,ls='--',c=colors_list[i],lw=1.6)

			axarr[1,j].axvline(x=core_radius[i_ar[j]],ls='-.',lw=1.5,color=colors_list[i],zorder=10*(i-1))
			if i==2: axarr[1,j].plot(drange,rho_nfw/rho_DM,lw=2.,c=colors_list[i],label='NFW/data',zorder=10*(i-1))		
			else: axarr[1,j].plot(drange,rho_nfw/rho_DM,lw=2.,c=colors_list[i],zorder=10*(i-2))

			print(sim,core_radius[i_ar[j]])
			
			if core_radius[i_ar[j]] > 0.05:
				text = axarr[1,j].annotate(r'$r_\mathregular{core}$ = '+str(int(np.round(core_radius[i_ar[j]]*1e3,0)))+' pc',
					xy=(core_radius[i_ar[j]]*1.05,hl[i-2]),fontsize=11,color=colors_list[i],zorder=1e4*(i-1))
				if not(sim=='vareff_1e6'):
					text.set_path_effects([path_effects.PathPatchEffect(linewidth=5, facecolor='black', edgecolor='white'), path_effects.Normal()])
			else:
				text = axarr[1,j].annotate(r'$r_\mathregular{core}$ = '+str(int(np.round(core_radius[i_ar[j]]*1e3,0)))+' pc',
					xy=(core_radius[i_ar[j]]+0.11,hl[i-2]),fontsize=11,color=colors_list[i],zorder=1e4*(i-1))
				if not(sim=='vareff_1e6'):
					text.set_path_effects([path_effects.PathPatchEffect(linewidth=5, facecolor='black', edgecolor='white'), path_effects.Normal()])

			#---plot-slope-------------------------------------------------------------------------
			if do_slope:
				sel = (drange < core_radius[i_ar[j]])
				drange_fit = drange[sel]
				rho_dm_fit = rho_DM[sel]

				slopecolor = 'gold'

				fitted_pars = fit.powerlaw_sqfit(xdata=drange_fit,ydata=rho_dm_fit,a_guess=1e8,k_guess=-0.1)
				# powerlaw = fit.powerlaw_model(fitted_pars,drange_fit)
				points = np.logspace(np.log10(0.11),np.log10(core_radius[i_ar[j]]),3)
				plp = fit.powerlaw_model(fitted_pars,points)
				axarr[0,j].plot(points,plp,'-s',lw=1.5,mew=0,ms=5,c=slopecolor)
				axarr[0,j].annotate(r'$\alpha$ = '+str(np.round(fitted_pars[1],2)),xy=(0.5,0.35-(0.05*i)),xycoords='axes fraction',fontsize=12,color=colors_list[i],alpha=1)
				if i==3: axarr[0,1].annotate('power law fit', xy=(0.65, 0.73), xycoords='axes fraction', fontsize=12,color=slopecolor,zorder=1e4)

			#---plot-setup-to-do-only-once---------------------------------------------------------
			if i==2:
				axarr[0,j].plot(drange,dark_rho_0,'-',lw=1,c='grey')
				axarr[1,j].axhline(y=2,color='grey',ls=':',lw=0.9)
				axarr[0,j].set_xscale('log')
				axarr[0,j].set_xlim(1e-1,1e1)
				snapfile = d.smuggledir + sim + '/snapshot_'+str(i_ar[j]).zfill(3)
				header = snapHDF5.snapshot_header(snapfile)
				time = header.time/0.7
				text = axarr[0,j].annotate(r'$t=$'+str(np.round(time,2))+' Gyr',xy=(0.05,0.92),
					xycoords='axes fraction',fontsize=11,color='black',zorder=1e4)
				text.set_path_effects([path_effects.PathPatchEffect(linewidth=5, facecolor='black', edgecolor='white'), path_effects.Normal()])

	#---finalize-plot------------------------------------------------------------------------------
	# axarr[0,1].legend(loc='upper right',frameon=False,prop={'size':11})
	# axarr[1,0].legend(loc='upper right',frameon=False,prop={'size':11})

	axarr[0,1].annotate('initial', xy=(0.65, 0.93-0.05*(5-2)), xycoords='axes fraction', fontsize=12,color='grey')
	axarr[0,0].annotate(r'solid: simulated $\rho_\mathregular{DM}$', xy=(0.5, 0.93), xycoords='axes fraction', fontsize=12,color='black')
	axarr[0,0].annotate(r'dashed: NFW fit', xy=(0.5, 0.88), xycoords='axes fraction', fontsize=12,color='black')

	axarr[1,0].annotate('NFW/data',xy=(0.75,0.82),xycoords='axes fraction',fontsize=11,color='black')

	axarr[0,0].set_yscale('log')
	axarr[0,0].set_ylabel(r'$\rho_\mathregular{DM}$ [M$_\odot$ kpc$^{-3}$]')
	axarr[0,0].set_ylim(1e6,3e9)
	axarr[0,0].set_yticks([1e6,1e7,1e8,1e9])
	# axarr[0,0].set_yticklabels(['',r'',1e7,1e8,1e9])

	axarr[1,0].set_ylabel(r'$\rho_\mathregular{NFW}$ / $\rho_\mathregular{DM}$')
	axarr[1,0].set_yscale('log')
	axarr[1,0].set_ylim(0.5,5)
	axarr[1,0].set_yticks([0.5,1,2])
	axarr[1,0].set_yticklabels([0.5,1,2])

	axarr[1,0].set_xlabel('distance [kpc]')

	axarr[0,0].set_xticks([1e-1,1e0]);		axarr[0,0].set_xticklabels(['0.1','1'])
	axarr[0,1].set_xticks([1e-1,1e0]);		axarr[0,1].set_xticklabels(['0.1','1'])
	axarr[0,2].set_xticks([1e-1,1e0]);		axarr[0,2].set_xticklabels(['0.1','1'])
	axarr[0,3].set_xticks([1e-1,1e0,1e1]);	axarr[0,3].set_xticklabels(['0.1','1','10'])

	# p.finalize(fig,'four_rho_'+sim,save=1,save_pdf=False)
	# fname = d.plotdir+'06.jun/four_rho_'+sim+'.png'
	# print('saving figure: '+fname)
	# plt.savefig(fname,format='png',dpi=200)
	p.finalize(fig,'four_rho_ratio_slope_fixMorph',save=save,save_pdf=save_pdf)

def four_rho_plots_ratio_set():
	if whichsims=='fixMorph_1e6':
		four_rho_plots_ratio_fixMorph()
	elif whichsims=='ff_ill_1e6':
		four_rho_plots_ratio_ffill()
	else:
		raise ValueError('four_rho_plots_ratio is not defined for '+whichsims+'. please choose either ff_ill_1e6 or fixMorph_1e6.')

def four_rho_plots_ratio(nfw_dcut=3,do_slope=1):
	fig, axarr = p.makefig(n_panels='4_horiz_with_ratio',figx=20,figy=6)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for i,sim in enumerate(models):
		text = axarr[0,1].annotate(models_label[i], xy=(0.65, 0.93-0.05*i), xycoords='axes fraction', fontsize=12,color=colors_list[i],zorder=1e4)

		print(sim)
		try: f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		except: f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		dark_mass_all = np.array(f['dark'])
		f.close()
		vols = 4./3.*np.pi*(drange**3)

		#---read-all-core-radius-----------------------------------------------------------------------
		try: nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'_hires.hdf5','r')
		except: nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'.hdf5','r')
		time = np.array(nf['time'])
		core_radius = np.array(nf['core_radius'])
		nf.close()

		pfname = d.datdir+'power_radius_'+sim+'.hdf5'
		pf = h5py.File(pfname,'r')
		power_radius = np.array(pf['power_radius'])
		pf.close()

		#---plot-density-profiles----------------------------------------------------------------------
		ilast = 400; imax = 227; inb = 169; imin = 65
		i_ar = np.array([imin,inb,imax,ilast])

		dark_rho_0 = dark_mass_all[0]/vols
		dark_rho_ar = np.vstack((dark_mass_all[imin]/vols, dark_mass_all[inb]/vols, dark_mass_all[imax]/vols, dark_mass_all[ilast]/vols))

		hl = [2.5,0.8,0.6]
		
		for j in np.arange(4):
			#--fit-NFW---------------------------------------------------------------------------------
			rho_DM = dark_rho_ar[j]
			H = 70./1000.
			rho_crit = 3*H**2 / (8*np.pi*m.Gprime) 
			r200 = drange[(rho_DM >= 200*rho_crit)][-1]
			rho_DM_fit = rho_DM[(drange > nfw_dcut)]
			drange_fit = drange[(drange > nfw_dcut)]

			fitted_pars = fit.NFW_sqfit(drange_fit,rho_DM_fit,c_guess=10,rho0_guess=1e7,r200=r200)
			rho_nfw = fit.NFW_model(fitted_pars,drange,r200)

			#--plot-the-stuff----------------------------------------------------------------------
			axarr[0,j].plot(drange,rho_DM,lw=3.5,c=colors_list[i],zorder=10*(i))
			axarr[0,j].axvline(x=core_radius[i_ar[j]],ls='-.',lw=1.5,color=colors_list[i],zorder=10*(i))
			axarr[0,j].plot(drange,rho_nfw,ls='--',c=colors_list[i],lw=1.6,zorder=10*(i))

			axarr[1,j].axvline(x=core_radius[i_ar[j]],ls='-.',lw=1.5,color=colors_list[i],zorder=10*(i))
			if i==2: axarr[1,j].plot(drange,rho_nfw/rho_DM,lw=2.,c=colors_list[i],label='NFW/data',zorder=10*(i))		
			else: axarr[1,j].plot(drange,rho_nfw/rho_DM,lw=2.,c=colors_list[i],zorder=10*(i))

			# print(sim,core_radius[i_ar[j]])
			
			if core_radius[i_ar[j]] > 0.05:
				text = axarr[1,j].annotate(r'$r_\mathregular{core}$ = '+str(int(np.round(core_radius[i_ar[j]]*1e3,0)))+' pc',
					xy=(core_radius[i_ar[j]]*1.05,hl[i]),fontsize=11,color=colors_list[i],zorder=1e4*(i+1))
				# if not(sim=='vareff_1e6'):
				text.set_path_effects([path_effects.PathPatchEffect(linewidth=5, facecolor='black', edgecolor='white'), path_effects.Normal()])
			else:
				text = axarr[1,j].annotate(r'$r_\mathregular{core}$ = '+str(int(np.round(core_radius[i_ar[j]]*1e3,0)))+' pc',
					xy=(core_radius[i_ar[j]]+0.11,hl[i]),fontsize=11,color=colors_list[i],zorder=1e4*(i+1))
				# if not(sim=='vareff_1e6'):
				text.set_path_effects([path_effects.PathPatchEffect(linewidth=5, facecolor='black', edgecolor='white'), path_effects.Normal()])

			#---plot-slope-------------------------------------------------------------------------
			if do_slope:
				sel = (drange < core_radius[i_ar[j]]) & (drange > power_radius[i_ar[j]])
				drange_fit = drange[sel]
				rho_dm_fit = rho_DM[sel]

				slopecolor = 'yellow'

				fitted_pars = fit.powerlaw_sqfit(xdata=drange_fit,ydata=rho_dm_fit,a_guess=1e8,k_guess=-0.1)
				# powerlaw = fit.powerlaw_model(fitted_pars,drange_fit)
				points = np.logspace(np.log10(0.11),np.log10(core_radius[i_ar[j]]),3)
				plp = fit.powerlaw_model(fitted_pars,points)
				axarr[0,j].plot(points,plp,'-s',lw=1.5,mew=0,ms=5,c=slopecolor,zorder=1e4*(i+1))
				axarr[0,j].annotate(r'$\alpha$ = '+str(np.round(fitted_pars[1],2)),xy=(0.5,0.2-(0.05*i)),xycoords='axes fraction',fontsize=12,color=colors_list[i],alpha=1)
				if i==3: axarr[0,1].annotate('power law fit', xy=(0.65, 0.73), xycoords='axes fraction', fontsize=12,color=slopecolor,zorder=1e4)

			#---plot-setup-to-do-only-once---------------------------------------------------------
			if i==0:
				axarr[0,j].plot(drange,dark_rho_0,'-',lw=1,c='grey')
				axarr[1,j].axhline(y=2,color='grey',ls=':',lw=0.9)
				axarr[0,j].set_xscale('log')
				axarr[0,j].set_xlim(1e-1,1e1)
				snapfile = d.smuggledir + sim + '/snapshot_'+str(i_ar[j]).zfill(3)
				header = snapHDF5.snapshot_header(snapfile)
				time = header.time/0.7
				text = axarr[0,j].annotate(r'$t=$'+str(np.round(time,2))+' Gyr',xy=(0.05,0.92),
					xycoords='axes fraction',fontsize=11,color='black',zorder=1e4)
				text.set_path_effects([path_effects.PathPatchEffect(linewidth=5, facecolor='black', edgecolor='white'), path_effects.Normal()])

	#---finalize-plot------------------------------------------------------------------------------

	axarr[0,1].annotate('initial', xy=(0.65, 0.93-0.05*(5-2)), xycoords='axes fraction', fontsize=12,color='grey')
	axarr[0,0].annotate(r'solid: simulated $\rho_\mathregular{DM}$', xy=(0.5, 0.93), xycoords='axes fraction', fontsize=12,color='black')
	axarr[0,0].annotate(r'dashed: NFW fit', xy=(0.5, 0.88), xycoords='axes fraction', fontsize=12,color='black')

	axarr[1,0].annotate('NFW/data',xy=(0.75,0.82),xycoords='axes fraction',fontsize=11,color='black')

	axarr[0,0].set_yscale('log')
	axarr[0,0].set_ylabel(r'$\rho_\mathregular{DM}$ [M$_\odot$ kpc$^{-3}$]')
	axarr[0,0].set_ylim(1e6,3e9)
	axarr[0,0].set_yticks([1e6,1e7,1e8,1e9])
	# axarr[0,0].set_yticklabels(['',r'',1e7,1e8,1e9])

	axarr[1,0].set_ylabel(r'$\rho_\mathregular{NFW}$ / $\rho_\mathregular{DM}$')
	axarr[1,0].set_yscale('log')
	axarr[1,0].set_ylim(0.5,5)
	axarr[1,0].set_yticks([0.5,1,2])
	axarr[1,0].set_yticklabels([0.5,1,2])

	axarr[1,0].set_xlabel('distance [kpc]')

	axarr[0,0].set_xticks([1e-1,1e0]);		axarr[0,0].set_xticklabels(['0.1','1'])
	axarr[0,1].set_xticks([1e-1,1e0]);		axarr[0,1].set_xticklabels(['0.1','1'])
	axarr[0,2].set_xticks([1e-1,1e0]);		axarr[0,2].set_xticklabels(['0.1','1'])
	axarr[0,3].set_xticks([1e-1,1e0,1e1]);	axarr[0,3].set_xticklabels(['0.1','1','10'])

	p.finalize(fig,'four_rho_ratio_'+whichsims,save=save,save_pdf=save_pdf)

def frac_rcore_time():
	print('plotting rho_frac_rcore_time')
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for i,sim in enumerate(models):
		print(sim)
		ax.annotate(models_label[i],xy=(0.75,0.2-(0.04*i)),xycoords='axes fraction',fontsize=11,color=colors_list[i])#,alpha=alphas_list[i])
		
		f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		drange = np.array(f['drange'])
		vols = 4./3.*np.pi*(drange**3)
		dark_profile_all = np.array(f['dark'])
		gas_profile_all = np.array(f['gas'])
		type2_profile_all = np.array(f['type2'])
		type3_profile_all = np.array(f['type3'])
		type4_profile_all = np.array(f['type4'])
		f.close()

		nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3_hires.hdf5','r')
		time = np.array(nf['time'])
		core_radius_all = np.array(nf['core_radius'])
		nf.close()

		max_snap = dark_profile_all.shape[0] 

		# rhofrac = np.array([])
		m_frac_all = np.array([])

		for snapnum in np.arange(0,max_snap):
			# rho_dm = dark_profile_all[snapnum] / vols
			# rho_gas = gas_profile_all[snapnum] / vols
			# rho_type2 = type2_profile_all[snapnum] / vols
			# rho_type3 = type3_profile_all[snapnum] / vols
			# rho_type4 = type4_profile_all[snapnum] / vols
			# rho_bary = rho_gas + rho_type2 + rho_type3 + rho_type4

			m_dm = dark_profile_all[snapnum] 
			m_gas = gas_profile_all[snapnum] 
			m_type2 = type2_profile_all[snapnum] 
			m_type3 = type3_profile_all[snapnum] 
			m_type4 = type4_profile_all[snapnum] 
			m_bary = m_gas + m_type2 + m_type3 + m_type4
			rcore = core_radius_all[snapnum]

			icore = m.find_nearest(drange,rcore,getindex=1)
			m_frac = m_bary[icore] / m_dm[icore]
			m_frac_all = np.append(m_frac_all,m_frac)

		ax.plot(time,m_frac_all,c=colors_list[i],lw=1.3,alpha=1,zorder=1000./(i+1.))

		# dt = 0.05
		# timebins = np.arange(0,np.amax(time)+dt,dt)
		# binwidth = (timebins[1] - timebins[0])/2
		# slope_mean = np.array([])

		# for j in range(len(timebins)-1):
		# 	leftbin = timebins[j]
		# 	rightbin = timebins[j+1]

		# 	sel = (time > leftbin) & (time < rightbin)
		# 	slope_mean = np.append(slope_mean, np.mean(slope[sel]))

		# ax.plot(timebins[0:-1]+binwidth,slope_mean,c=colors_list[i],lw=2,alpha=1,zorder=1000./(i+1.)+10)

		# meanslope = np.mean(slope[(time > 0.75)])
		# ax.axhline(y=meanslope,c=colors_list[i],lw=1,ls=':')


	ax.set_xlabel('time [Gyr]')
	ax.set_xlim(0,2.86)
	ax.set_xticks([0,0.5,1,1.5,2,2.5])
	ax.set_xticks([0.25,0.75,1.25,1.75,2.25,2.75],minor=True)

	ax.set_ylabel(r'$M_\mathregular{bary} / M_\mathregular{dm} (r_\mathregular{core})$')
	ax.set_yscale('log')
	ax.set_ylim(1e-1,2e0)


	p.finalize(fig,'frac_rcore_time_'+whichsims,save=save)

def get_masses():
	print('getting masses')
	for i,sim in enumerate(models):

		mypath = d.smuggledir+sim
		onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
		a = np.sort(onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)])
		max_snap = int(a[-1].split('.')[0][-3:])

		snapfile = d.smuggledir+sim+'/snapshot_'+str(max_snap).zfill(3)
		strmass = snapHDF5.read_block(snapfile,"MASS", parttype=4)*(1.e10)/m.h
		darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h

		print('------------------------------------------------------------')
		print(sim+'  snapshot '+str(max_snap))
		print('total type 4 (stellar) mass is: '+m.scinote(np.sum(strmass)))
		print('total type 1 (halo) mass is: '+m.scinote(np.sum(darkmass)))

def get_power_radius(sim,snapnum):
	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
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

	print('\ndistance of the 200th DM particle: ' + str(np.round(d_200,5)))

def get_r200(snapnum=400):
	for sim in models:
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		vols = 4./3.*np.pi*(drange**3)
		rho_DM = np.array(f['dark'])[snapnum] / vols
		f.close()

		H = 70./1000.
		rho_crit = 3*H**2 / (8*np.pi*m.Gprime) 
		r200 = drange[(rho_DM >= 200*rho_crit)][-1]

		print('\n-----------------------------------------\n'+sim)
		print('r200 = '+str(np.round(r200,5))+' kpc')


def is_particle_present(sim):
	print('plotting is_particle_present')
	print(sim)
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# get initial snapshot and pick particle at random
	snapfile = d.smuggledir+sim+'/snapshot_'+str(0).zfill(3)	
	dark_id_all = snapHDF5.read_block(snapfile, 'ID  ', parttype=1)
	trace_id = np.random.choice(dark_id_all)

	max_snap = 401

	# check that our tracer ID is present at all snapshots
	is_present = np.array([])
	for snapnum in np.arange(0,max_snap):
		printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
		sys.stdout.write(printthing)
		sys.stdout.flush()
		if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
		else: print('')

		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)	
		dark_id_all = snapHDF5.read_block(snapfile, 'ID  ', parttype=1)
		# if trace_id in dark_id_all:
		is_present = np.append(is_present, (trace_id in dark_id_all))

	num_present = np.count_nonzero(is_present)

	print('trace id is present for '+str(num_present)+'/400 snapshots')

def list_rcores(sim):
	f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
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

def maxrho_dist_time():
	print('plotting rho_frac_rcore_time')
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for i,sim in enumerate(models):
		print(sim)
		ax.annotate(models_label[i],xy=(0.05,0.95-(0.05*i)),xycoords='axes fraction',fontsize=11,color=colors_list[i])#,alpha=alphas_list[i])
		
		f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		drange = np.array(f['drange'])
		vols = 4./3.*np.pi*(drange**3)
		dark_profile_all = np.array(f['dark'])
		gas_profile_all = np.array(f['gas'])
		type2_profile_all = np.array(f['type2'])
		type3_profile_all = np.array(f['type3'])
		type4_profile_all = np.array(f['type4'])
		f.close()

		nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3'+'_hires.hdf5','r')
		time = np.array(nf['time'])
		nf.close()

		rmax_all = np.array([])
		for snapnum in np.arange(0,401):
			rho_dm = dark_profile_all[snapnum] / vols
			rho_gas = gas_profile_all[snapnum] / vols
			rho_type2 = type2_profile_all[snapnum] / vols
			rho_type3 = type3_profile_all[snapnum] / vols
			rho_type4 = type4_profile_all[snapnum] / vols
			rho_bary = rho_gas + rho_type2 + rho_type3 + rho_type4

			rho_frac = rho_bary/rho_dm		
			drange_nonans = drange[np.logical_not(np.isnan(rho_frac))]
			rho_frac = rho_frac[np.logical_not(np.isnan(rho_frac))] # remove nans

			is_equal = rho_frac==np.amax(rho_frac) # array of trues and falses
			is_equal_true_locs = is_equal.nonzero()[0] # array of locations where rho_frac is max - just numbers
			is_consecutive = all(p==1 for p in np.ediff1d(is_equal_true_locs))

			# if len(is_equal_true_locs) > 1:
			# 	print(is_equal_true_locs)
	
			imax = np.amax(is_equal_true_locs)
			rmax = drange_nonans[imax]
			rmax_all = np.append(rmax_all,rmax)

		# ax.plot(time,rmax_all,c=colors_list[i],lw=1,alpha=1)
		dt = 0.05
		timebins = np.arange(0,np.amax(time)+dt,dt)
		binwidth = (timebins[1] - timebins[0])/2
		rmax_all_mean = np.array([])

		for j in range(len(timebins)-1):
			leftbin = timebins[j]
			rightbin = timebins[j+1]

			sel = (time > leftbin) & (time < rightbin)
			rmax_all_mean = np.append(rmax_all_mean, np.mean(rmax_all[sel]))

		ax.plot(timebins[0:-1]+binwidth,rmax_all_mean,c=colors_list[i],lw=1.3,alpha=1,zorder=1000./(i+1.)+10)
	
	ax.set_xlabel('time [Gyr]')
	ax.set_xlim(0,2.86)
	ax.set_xticks([0,0.5,1,1.5,2,2.5])
	ax.set_xticks([0.25,0.75,1.25,1.75,2.25,2.75],minor=True)

	ax.set_ylabel(r'$r (\rho_\mathregular{bary} / \rho_\mathregular{dm} = \mathregular{max})$ [kpc]')
	ax.set_yscale('log')
	ax.set_ylim(0.01,0.1)
	ax.set_yticks([0.01,0.1,1])
	ax.set_yticklabels([0.01,0.1,1])

	p.finalize(fig,'maxrho_dist_time_'+whichsims,save=save)

def masstime(ptype,do_all=True):
	if len(models)==2:
		fig,axarr = p.makefig('4_by_2',figx=8,figy=8)
	# elif (whichsims=='fixMorph_1e6' or whichsims=='fixMorph_1e5') and not(do_all):
	# 	fig,axarr = p.makefig('4_by_3',figx=12,figy=8)
	elif len(models)==5:
		fig,axarr = p.makefig('4_by_5',figx=20,figy=8)

	else:
		raise ValueError('please choose appropriate whichsims')	

	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)

	if not ptype in np.array(['gas','dark','star','gasfrac','baryfrac']):
		raise ValueError('please choose an available ptype')

	dist_arr = np.array([0.1,0.5,1,2])
	# dist_arr = np.array([0.5,1,2,5])
	# if ptype=='dark':
	# 	dist_arr = np.array([0.1,0.5,1,2])
	# else:
	# 	dist_arr = np.array([0.5,1,2,5])

	for i,sim in enumerate(models):
		if whichsims=='fixMorph_1e6' and i<2 and not(do_all):
			continue
		print(sim)
		axarr[0,i].annotate(models_label[i],xy=(0.05,0.85),xycoords='axes fraction',size=11,color=colors_list[i])

		# f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		try:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		except:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')

		if ptype=='star':
			type2_profile = np.array(f['type2'])
			type3_profile = np.array(f['type3'])
			type4_profile = np.array(f['type4'])
			max_snap = type4_profile.shape[0]

		elif ptype=='gas' or ptype=='dark':
			massprofile = np.array(f[ptype])
			max_snap = massprofile.shape[0]

		elif ptype=='gasfrac':
			gas_profile = np.array(f['gas'])
			dark_profile = np.array(f['dark'])
			max_snap = gas_profile.shape[0]

		elif ptype=='baryfrac':
			gas_profile = np.array(f['gas'])
			dark_profile = np.array(f['dark'])
			type2_profile = np.array(f['type2'])
			type3_profile = np.array(f['type3'])
			type4_profile = np.array(f['type4'])
			max_snap = gas_profile.shape[0]

		drange = np.array(f['drange'])
		f.close()

		vols = 4./3.*np.pi*(drange**3)
		snaplist = np.arange(0,max_snap)
		time = snaplist/400.*2.857

		for j,dist in enumerate(dist_arr):
			sel_d = (drange <= dist)

			if ptype=='star':
				type2_in_d = np.array([]); type3_in_d = np.array([]); type4_in_d = np.array([])

				for snapnum in snaplist:
					m_type2 = type2_profile[snapnum]; m_type3 = type3_profile[snapnum]; m_type4 = type4_profile[snapnum]
					type2_in_d = np.append(type2_in_d,m_type2[sel_d][-1])
					type3_in_d = np.append(type3_in_d,m_type3[sel_d][-1])
					type4_in_d = np.append(type4_in_d,m_type4[sel_d][-1])

				if j==i==0:
					axarr[j,i].plot(time,type2_in_d,c=colors_list[i],ls=':',label='type 2')
					axarr[j,i].plot(time,type3_in_d,c=colors_list[i],ls='--',label='type 3')
					axarr[j,i].plot(time,type4_in_d,c=colors_list[i],ls='-',label='type 4')
				else:
					axarr[j,i].plot(time,type2_in_d,c=colors_list[i],ls=':')
					axarr[j,i].plot(time,type3_in_d,c=colors_list[i],ls='--')
					axarr[j,i].plot(time,type4_in_d,c=colors_list[i],ls='-')
			
			elif ptype=='gas' or ptype=='dark':
				rho_in_d_all = np.array([])
				for snapnum in snaplist:
					rho = massprofile[snapnum]#/vols #* 4.046e-8
					rho_in_d_all = np.append(rho_in_d_all,rho[sel_d][-1])
				axarr[j,i].plot(time,rho_in_d_all,c=colors_list[i])

			elif ptype=='gasfrac':
				gas_in_d = np.array([]); dark_in_d = np.array([])
				for snapnum in snaplist:
					m_gas = gas_profile[snapnum]; m_dark = dark_profile[snapnum];
					gas_in_d = np.append(gas_in_d,m_gas[sel_d][-1])
					dark_in_d = np.append(dark_in_d,m_dark[sel_d][-1])
				frac = gas_in_d/dark_in_d
				axarr[j,i].plot(time,frac,c=colors_list[i])

			elif ptype=='baryfrac':
				bary_in_d = np.array([]); dark_in_d = np.array([]); 
				for snapnum in snaplist:
					m_gas = gas_profile[snapnum]; m_dark = dark_profile[snapnum]; m_type2 = type2_profile[snapnum]; m_type3 = type3_profile[snapnum]; m_type4 = type4_profile[snapnum]
					m_bary = m_gas + m_type2 + m_type3 + m_type4
					bary_in_d = np.append(bary_in_d,m_bary[sel_d][-1])
					dark_in_d = np.append(dark_in_d,m_dark[sel_d][-1])
				frac = bary_in_d/dark_in_d
				axarr[j,i].plot(time,frac,c=colors_list[i])

			axarr[j,i].annotate(r'd < '+str(dist)+' kpc',xy=(0.7,0.15),xycoords='axes fraction',size=11,color='black')
			# if j<2:
			# 	axarr[j,i].annotate(r'd < '+str(dist)+' kpc',xy=(0.7,0.15),xycoords='axes fraction',size=11,color='black')
			# else:
			# 	# axarr[j,i].annotate(r'd < '+str(dist)+' kpc',xy=(0.7,0.15),xycoords='axes fraction',size=11,color='black')
			# 	axarr[j,i].annotate(r'd < '+str(dist)+' kpc',xy=(0.7,0.85),xycoords='axes fraction',size=11,color='black')
			axarr[j,i].set_yscale('log')

	axarr[3,0].set_xlim(0,2.857)
	axarr[3,int(axarr.shape[1])/2.].set_xlabel('time [Gyr]')
	axarr[0,int(axarr.shape[1])/2.].set_title('type = '+ptype,size=15)
	# axarr[1,0].set_ylabel(r'$\rho_\mathregular{gas}$ [M$_\odot$ kpc$^{-3}$]'0

	if ptype=='gas':
		axarr[1,0].set_ylabel(r'$M_\mathregular{gas}$ [M$_\odot$]')
		axarr[0,0].set_ylim(1e4,1e7)
		axarr[0,0].set_yticks([1e4,1e5,1e6,1e7])
		axarr[1,0].set_ylim(1e6,1e9)
		axarr[1,0].set_yticks([1e6,1e7,1e8])
		axarr[2,0].set_ylim(1e6,1e9)
		axarr[2,0].set_yticks([1e6,1e7,1e8])
		axarr[3,0].set_ylim(1e6,1e9)
		axarr[3,0].set_yticks([1e6,1e7,1e8])
	elif ptype=='dark':
		axarr[1,0].set_ylabel(r'$M_\mathregular{DM}$ [M$_\odot$]')
		axarr[0,0].set_ylim(1e5,1e7)
		axarr[0,0].set_yticks([1e5,1e6,1e7])
		axarr[1,0].set_ylim(1e7,1e9)
		axarr[1,0].set_yticks([1e7,1e8])
		axarr[2,0].set_ylim(1e8,1e10)
		axarr[2,0].set_yticks([1e8,1e9])
		axarr[3,0].set_ylim(1e8,1e10)
		axarr[3,0].set_yticks([1e8,1e9])
	elif ptype=='star':
		axarr[0,0].legend(prop={'size':9},loc='upper center',frameon=False)
		axarr[1,0].set_ylabel(r'$M_\ast}$ [M$_\odot$]')
		axarr[0,0].set_ylim(1e6,3e8)
		axarr[0,0].set_yticks([1e6,1e7,1e8])
		axarr[1,0].set_ylim(1e6,3e8)
		axarr[1,0].set_yticks([1e6,1e7,1e8])
		axarr[2,0].set_ylim(1e6,3e8)
		axarr[2,0].set_yticks([1e6,1e7,1e8])
		axarr[3,0].set_ylim(1e6,3e8)
		axarr[3,0].set_yticks([1e6,1e7,1e8])
	elif ptype=='gasfrac':
		axarr[0,0].legend(prop={'size':9},loc='upper center',frameon=False)
		axarr[1,0].set_ylabel(r'$M_\mathregular{gas} / M_\mathregular{DM}$')
		axarr[0,0].set_ylim(1e-2,1e0)
		axarr[0,0].set_yticks([1e-2,1e-1,1e0])
		axarr[1,0].set_ylim(1e-2,1e0)
		axarr[1,0].set_yticks([1e-2,1e-1])
		axarr[2,0].set_ylim(1e-2,1e0)
		axarr[2,0].set_yticks([1e-2,1e-1])
		axarr[3,0].set_ylim(1e-2,1e0)
		axarr[3,0].set_yticks([1e-2,1e-1])
	elif ptype=='baryfrac':
		axarr[0,0].legend(prop={'size':9},loc='upper center',frameon=False)
		axarr[1,0].set_ylabel(r'$M_\mathregular{baryon} / M_\mathregular{DM}$')
		axarr[0,0].set_ylim(5e-2,5e0)
		axarr[0,0].set_yticks([1e-1,1e0])
		axarr[1,0].set_ylim(1e-2,1e0)
		axarr[1,0].set_yticks([1e-2,1e-1])
		axarr[2,0].set_ylim(1e-2,1e0)
		axarr[2,0].set_yticks([1e-2,1e-1])
		axarr[3,0].set_ylim(1e-2,1e0)
		axarr[3,0].set_yticks([1e-2,1e-1])
	
	# if (whichsims=='ff_ill_1e6') or (whichsims=='fixMorph_1e6' and not(do_all)): fname = 'masstime_'+ptype+'_'+whichsims
	# elif  whichsims=='fixMorph_1e6' and do_all: fname='masstime_'+ptype+'_'+whichsims+'_all'
	# elif whichsims=='smuggle_1e6' or  whichsims=='smuggle_1e5': fname='masstime_'+ptype+'_'+whichsims

	fname='masstime_'+ptype+'_'+whichsims

	p.finalize(fig,fname,save=save,save_pdf=save_pdf)

def masstime_single(dist,ptype):
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()
	if ptype=='dark': 
		yfx = 0.95
		ax.annotate('dark matter  '+r'$r < $'+str(dist)+' kpc',xy=(0.1,0.9),xycoords='axes fraction',fontsize=12,color='black')	
	elif ptype=='type4':
		yfx = 0.3
		ax.annotate('stars  '+r'$r < $'+str(dist)+' kpc',xy=(0.1,0.9),xycoords='axes fraction',fontsize=12,color='black')	
	elif ptype=='gas':
		ax.annotate('gas  '+r'$r < $'+str(dist)+' kpc',xy=(0.1,0.9),xycoords='axes fraction',fontsize=12,color='black')	
	
	for i,sim in enumerate(models):
		print(sim)
		text = ax.annotate(models_label[i],xy=(0.8,yfx-(0.06*i)),xycoords='axes fraction',fontsize=12,color=colors_list[i])#,alpha=alphas_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=colors_list[i], edgecolor='white'), path_effects.Normal()])
		
		f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		drange = np.array(f['drange'])
		mass_profile = np.array(f[ptype])
		f.close()
		i_dist = m.find_nearest(drange,dist,getindex=True)

		m_in_dist = np.array([])
		for n in np.arange(401):
			m_this = mass_profile[n][i_dist]
			m_in_dist = np.append(m_in_dist,m_this)

		ax.plot(time,m_in_dist,c=colors_list[i],lw=1,alpha=1)

	ax.set_xlim(0,np.amax(time))
	# ax.set_xticks
	ax.set_xlabel('time [Gyr]')
	ax.set_yscale('log')
	if ptype=='dark': 
		ax.set_ylabel(r'$M_\mathregular{DM}$ ($r$ < '+str(dist)+r' kpc) [M$_\odot$]')
		ax.set_ylim(1e5,1e7)

	elif ptype=='type4': 
		ax.set_ylabel(r'$M_\ast$ ($r$ < '+str(dist)+r' kpc) [M$_\odot$]')
		ax.set_ylim(1e3,1e7)

	elif ptype=='gas': 
		ax.set_ylabel(r'$M_\mathregular{gas}$ ($r$ < '+str(dist)+r' kpc) [M$_\odot$]')
		ax.set_ylim(1e5,1e9)

	p.finalize(fig,'masstime_'+ptype+'_'+whichsims,save=save)

def measure_vmax(sim,snapnum,ptype):

	f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
	drange = np.array(f['drange'])
	vols = 4./3.*np.pi*(drange**3)

	gas_profile = np.array(f['gas'])[snapnum]
	dark_profile = np.array(f['dark'])[snapnum]
	type2_profile = np.array(f['type2'])[snapnum]
	type3_profile = np.array(f['type3'])[snapnum]
	type4_profile = np.array(f['type4'])[snapnum]
	f.close()

	if ptype=='dark':
		total_profile = dark_profile


	vcirc = np.sqrt(m.Gprime*total_profile/drange)
 
	print('vmax for '+sim+' at snapshot '+str(snapnum) + ' is: '+str(np.amax(vcirc)))

def multi_sfr_aperture_cumulative(do_bins=0):
	print('plotting sfr_aperture')
	fig,axarr = p.makefig('4_vert',figx=6.67,figy=12)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	ap_list = np.array([0.1,0.2,0.5,1])

	for i,sim in enumerate(models):
		f = h5py.File(d.datdir+'SFRprofile_'+sim+'.hdf5','r')
		SFR_profile_all = np.array(f['SFR_profile_all'])
		drange = np.array(f['drange'])
		f.close()
		axarr[0].annotate(models_label[i],xy=(0.05,0.92-(i*0.07)),xycoords='axes fraction',color=colors_list[i],size=12)

		for j,aperture in enumerate(ap_list):

			i_ap = m.find_nearest(drange,aperture,getindex=1)

			sfr_plt = np.array([])
			for snapnum in np.arange(0,SFR_profile_all.shape[0]):
				this_sfr_profile = SFR_profile_all[snapnum]
				sfr_plt = np.append(sfr_plt,this_sfr_profile[i_ap])

			axarr[j].plot(time,sfr_plt,lw=1,c=colors_list[i])


	axarr[3].set_xlim(0,2.86)
	axarr[3].set_xlabel('time [Gyr]')

	for j in np.array([0,1,2,3]):
		axarr[j].set_yscale('log')
		axarr[j].set_ylim(1e-4,1e0)
		axarr[j].annotate(r'$r < $'+str(ap_list[j])+' kpc',xy=(0.75,0.85),xycoords='axes fraction',color='black',size=15)

	axarr[1].set_ylabel(r'SFR' + r' [M$_\odot$ yr$^{-1}$]')	
	axarr[0].set_yticks([1e-4,1e-3,1e-2,1e-1,1e0])
	axarr[1].set_yticks([1e-4,1e-3,1e-2,1e-1])
	axarr[2].set_yticks([1e-4,1e-3,1e-2,1e-1])
	axarr[3].set_yticks([1e-4,1e-3,1e-2,1e-1])

	p.finalize(fig,'multi_sfr_aperture_'+whichsims,save=save)

def multi_sfr_shells(do_bins=0):
	print('plotting multi_sfr_shells')
	fig,axarr = p.makefig('6_vert',figx=6.67,figy=15)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	# ap_list = np.array([0.1,0.2,0.5,1])
	bins = np.array([0,0.1,0.3,0.5,1,3,np.inf])

	for i,sim in enumerate(models):
		axarr[0].annotate(models_label[i],xy=(0.05,0.92-(i*0.1)),xycoords='axes fraction',color=colors_list[i],size=12)
		print(sim)

		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()
		max_snap = CoM_all.shape[0]-1

		for snapnum in np.arange(0,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing); sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')
			snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
			gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h - CoM_all[snapnum]
			gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)

			gas_pos = gas_pos[(gas_sfr > 0)]
			gas_sfr = gas_sfr[(gas_sfr > 0)]

			r_gas = np.linalg.norm(gas_pos,axis=1) 
			
			sfr_arr_this = np.array([])
			for j in np.arange(len(bins)-1):
				sel = (r_gas >= bins[j]) & (r_gas < bins[j+1])
				sfr_arr_this = np.append(sfr_arr_this,np.sum(gas_sfr[sel]))

			if snapnum == 0:
				sfr_arr = sfr_arr_this
			else:
				sfr_arr = np.vstack((sfr_arr,sfr_arr_this))
		# print(sfr_arr.shape)
		# print(time.shape)
		for j in np.arange(sfr_arr.shape[1]):
			# print(time[:max_snap].shape)
			axarr[j].plot(time[:max_snap+1],sfr_arr[:,j],lw=1,c=colors_list[i])


	axarr[5].set_xlim(0,2.86)
	axarr[5].set_xlabel('time [Gyr]')

	for j in np.arange(len(bins)-1):
		axarr[j].set_yscale('log')
		axarr[j].set_ylim(1e-4,1e0)
		axarr[j].annotate(str(bins[j])+r' < $r$/kpc < '+str(bins[j+1]),xy=(0.55,0.85),xycoords='axes fraction',color='black',size=15)

	axarr[1].set_ylabel(r'SFR' + r' [M$_\odot$ yr$^{-1}$]')	
	axarr[0].set_yticks([1e-4,1e-3,1e-2,1e-1,1e0])
	axarr[1].set_yticks([1e-4,1e-3,1e-2,1e-1])
	axarr[2].set_yticks([1e-4,1e-3,1e-2,1e-1])
	axarr[3].set_yticks([1e-4,1e-3,1e-2,1e-1])
	axarr[4].set_yticks([1e-4,1e-3,1e-2,1e-1])
	axarr[5].set_yticks([1e-4,1e-3,1e-2,1e-1])

	p.finalize(fig,'multi_sfr_shells_'+whichsims,save=save)

def M_v_time(sim):
	print('plotting M_v_time')
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	try:
		f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
	except:
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')

	drange = np.array(f['drange'])
	vols = 4./3.*np.pi*(drange**3)

	gas_profile_all = np.array(f['gas'])
	dark_profile_all = np.array(f['dark'])
	type2_profile_all = np.array(f['type2'])
	type3_profile_all = np.array(f['type3'])
	type4_profile_all = np.array(f['type4'])
	f.close()

	max_snap = gas_profile_all.shape[0]

	dist_arr = np.array([0.5,1,2,5])

	for dist in dist_arr:

		mass = np.array([])
		idist = m.find_nearest(drange,dist,getindex=1)

		for snapnum in np.arange(max_snap):
			this_mass_profile = gas_profile_all[snapnum]
			mass = np.append(mass,this_mass_profile[idist])

		ax.plot(time[:max_snap],mass)

	ax.set_xlim(0,np.amax(time))
	ax.set_xlabel('time [Gyr]')

	ax.set_ylabel(r'$M$ [M$_\odot$]')
	ax.set_yscale('log')

	p.finalize(fig,'m_time_'+sim,save=save)

def NFW_fit(sim,snapnum):
	print('plotting NFW_fit')
	fig,ax = p.makefig(1)
	#--read-simulation-data------------------------------------------------------------------------
	# datdir = '/home/ethan/research/data/hdf5/'
	f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	vols = 4./3.*np.pi*(drange**3)
	rho_DM = np.array(f['dark'])[snapnum] / vols
	type4_profile = np.array(f['type4'])[snapnum]
	f.close()

	ax.plot(drange,rho_DM,'k',label='simulation',lw=2)#r'$\rho_\mathregular{DM}$')

	#--fit-NFW-------------------------------------------------------------------------------------
	H = 70./1000.
	rho_crit = 3*H**2 / (8*np.pi*m.Gprime) 
	r200 = drange[(rho_DM >= 200*rho_crit)][-1]
	rho_DM_fit = rho_DM[(drange > 10)]
	drange_fit = drange[(drange > 10)]

	fitted_pars = fit.NFW_sqfit(drange_fit,rho_DM_fit,c_guess=10,rho0_guess=1e7,r200=r200)
	rho_nfw = fit.NFW_model(fitted_pars,drange,r200)
	ax.plot(drange,rho_nfw,ls='--',c='b',label='NFW',lw=1.6)

	#--fit-coreNFW---------------------------------------------------------------------------------
	c = fitted_pars[0]
	rho0 = fitted_pars[1]

	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
	# Mstr_total = np.sum(snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h)
	# # Mstr_total = 7.e7
	# ihalf = m.find_nearest(type4_profile, Mstr_total/2., getindex=True)
	# Rhalf = drange[ihalf]
	# Rhalf = 10

	header = snapHDF5.snapshot_header(snapfile)
	# tsf = 1.5
	tsf = header.time

	fitted_pars_core = fit.coreNFW_sqfit(drange, rho_DM, 0.04, 0.5, r200, c, rho0, tsf)

	rho_coreNFW = fit.coreNFW_model(fitted_pars_core, drange, r200, c, rho0, tsf)

	ax.plot(drange,rho_coreNFW,ls='--',c='g',label='coreNFW',lw=1.6)


	#--set-up-plot---------------------------------------------------------------------------------

	ax.annotate('c = '+str(np.round(fitted_pars[0],2)),xy=(0.5,0.95),xycoords='axes fraction',fontsize=10,color='black')
	ax.annotate(r'$\rho_0$ = '+m.scinote(fitted_pars[1]),xy=(0.5,0.92),xycoords='axes fraction',fontsize=10,color='black')
	ax.annotate(r'$\kappa$ = '+str(np.round(fitted_pars_core[0],2)),xy=(0.5,0.89),xycoords='axes fraction',fontsize=10,color='black')
	ax.annotate(r'r$_c$ = '+str(np.round(fitted_pars_core[1],2)),xy=(0.5,0.86),xycoords='axes fraction',fontsize=10,color='black')

	ax.annotate(sim,xy=(0.05,0.15),xycoords='axes fraction',fontsize=10,color='black')
	ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.05,0.12),xycoords='axes fraction',fontsize=10,color='black')


	ax.axvline(x=fitted_pars_core[1],c='purple',ls=':',lw=1,label='core radius')

	ax.legend(prop={'size':10},frameon=False)

	ax.set_xscale('log')
	ax.set_xlim(0.1,100)
	ax.set_xlabel(r'r [kpc]')

	ax.set_yscale('log')
	ax.set_ylim(1e4,1e9)
	ax.set_ylabel(r'$\rho$ [M$_\odot$ kpc$^{-3}$]')

	p.finalize(fig,'nfw_fit_'+sim+'_'+str(snapnum).zfill(3),save=save)

def panel_projection_single(sim,snapnum,bound=5,dark_background=True,show_progress=True,do_rhalf=False,do_rcore=False,do_scale=True,do_analysis=False):
	# print('plotting panel_projection_single')
	if sim in models:
		i = np.where(models==sim)[0][0]
		thislabel = models_label[i]
	else:
		raise ValueError('please choose a sim in models')

	savedir = '/home/ejahn003/movie_frames/'+sim+'/'
	# savedir = '/home/ejahn003/plots/'+month+'/'+sim

	fname = d.smuggledir+sim+'/snapshot_' + str(snapnum).zfill(3)

	if do_analysis:	fig, axarr = p.makefig(n_panels='2_proj_with_ticks')
	else:			fig, axarr = p.makefig(n_panels='2_proj')

	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	if dark_background:	
		plt.style.use('dark_background')
		txtclr = 'white'
		strclr = 'orange'
	else:
		txtclr = 'black'
		strclr = 'darkorange'

	

	#---------------------------------------------
	gas_pos = snapHDF5.read_block(fname,"POS ",parttype=0)/h
	gasmass = snapHDF5.read_block(fname,"MASS", parttype=0)*(1.e10)/h
	str_pos = snapHDF5.read_block(fname,"POS ",parttype=4)/h
	strmass = snapHDF5.read_block(fname,"MASS", parttype=4)*(1.e10)/h
	darkpos = snapHDF5.read_block(fname,"POS ",parttype=1)/h
	darkmass = snapHDF5.read_block(fname,"MASS", parttype=1)*(1.e10)/h

	#---------------------------------------------
	# x_cm = np.sum(darkpos[:,0] * darkmass) / np.sum(darkmass)
	# y_cm = np.sum(darkpos[:,1] * darkmass) / np.sum(darkmass)
	# z_cm = np.sum(darkpos[:,2] * darkmass) / np.sum(darkmass)
	# cm = np.array([x_cm,y_cm,z_cm]).T
	cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
	cm = np.array(cm_file['CoM'])[snapnum]
	cm_file.close()

	darkpos = darkpos - cm
	str_pos = str_pos - cm
	gas_pos = gas_pos - cm

	#---------------------------------------------
	if '1e6' in sim:
		gas_ap = 0.3
		star_ap = 0.3
	elif '1e5' in sim:
		gas_ap = 0.6
		star_ap = 1

	#---plot-face-on---
	sel_gas_front = (gas_pos[:,2] > 0)
	sel_gas_back = (gas_pos[:,2] <= 0)
	
	axarr[0].plot(gas_pos[:,0][sel_gas_front],gas_pos[:,1][sel_gas_front],',',c='blue',alpha=gas_ap,zorder=100)
	if np.sum(strmass) > 0:
		axarr[0].plot(str_pos[:,0],str_pos[:,1],',',c=strclr,alpha=star_ap,zorder=10)
	axarr[0].plot(gas_pos[:,0][sel_gas_back],gas_pos[:,1][sel_gas_back],',',c='blue',alpha=gas_ap,zorder=1)	
	
	#---plot-edge-on---
	sel_gas_front = (gas_pos[:,1] > 0)
	sel_gas_back = (gas_pos[:,1] <= 0)	

	axarr[1].plot(gas_pos[:,0][sel_gas_front],gas_pos[:,2][sel_gas_front],',',c='blue',alpha=0.1,zorder=100)	
	if np.sum(strmass) > 0:
		axarr[1].plot(str_pos[:,0],str_pos[:,2],',',c=strclr,alpha=star_ap,zorder=10)
	axarr[1].plot(gas_pos[:,0][sel_gas_back],gas_pos[:,2][sel_gas_back],',',c='blue',alpha=gas_ap,zorder=1)

	height = 0.88

	if do_rhalf:
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
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
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
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
	

	if do_analysis:
		xl=0.75; xh=0.9; yl=-4.9; yh=-4.7; zl=-1.3; zh=-1.1
		sel_part = (str_pos[:,0] > xl) & (str_pos[:,0] < xh) & (str_pos[:,1] > yl) & (str_pos[:,1] < yh) & (str_pos[:,2] > zl) & (str_pos[:,2] < zh)
		strmass_part = strmass[sel_part]
		m_part = np.sum(strmass_part)


		str_pos_part = str_pos[sel_part]
		x_cm = np.sum(str_pos_part[:,0] * strmass_part) / np.sum(strmass_part)
		y_cm = np.sum(str_pos_part[:,1] * strmass_part) / np.sum(strmass_part)
		z_cm = np.sum(str_pos_part[:,2] * strmass_part) / np.sum(strmass_part)
		cm = np.array([x_cm,y_cm,z_cm]).T
		cm_str = str(np.round(cm[0],2))+', '+str(np.round(cm[1],2))+', '+str(np.round(cm[2],2))
		

		star_vel = snapHDF5.read_block(fname, 'VEL ', parttype=4)
		star_vel_part = star_vel[sel_part]
		vx_cm = np.sum(star_vel_part[:,0] * strmass_part) / np.sum(strmass_part)
		vy_cm = np.sum(star_vel_part[:,1] * strmass_part) / np.sum(strmass_part)
		vz_cm = np.sum(star_vel_part[:,2] * strmass_part) / np.sum(strmass_part)
		vcm = np.array([vx_cm,vy_cm,vz_cm]).T
		vcm_hat = (vcm / np.sqrt(vx_cm**2 + vy_cm**2 + vz_cm**2))*2
		vcm_str = str(np.round(vcm[0]))+', '+str(np.round(vcm[1]))+', '+str(np.round(vcm[2]))

		axarr[0].arrow(x_cm,y_cm,vcm_hat[0],vcm_hat[1],color='green',width=0.02,zorder=1000)
		axarr[1].arrow(x_cm,z_cm,vcm_hat[0],vcm_hat[2],color='green',width=0.02,zorder=1000)

		axarr[0].annotate(r'M$_\ast$ = '+m.scinote(m_part)+r'M$_\odot$', 
			xy=(0.05, 0.88), xycoords='axes fraction', fontsize=10,color=txtclr)

		axarr[0].annotate(r'$\mathit{\vec{r}}_\mathregular{cm}$ = '+cm_str+' kpc', 
			xy=(0.05, 0.84), xycoords='axes fraction', fontsize=10,color=txtclr)
		axarr[0].plot(cm[0],cm[1],'o',mew=0,mfc='red',ms=4,zorder=1000)
		axarr[1].plot(cm[0],cm[2],'o',mew=0,mfc='red',ms=4,zorder=1000)

		axarr[0].arrow(x_cm,y_cm,vcm_hat[0],vcm_hat[1],color='green',width=0.0005,zorder=1000)
		axarr[1].arrow(x_cm,z_cm,vcm_hat[0],vcm_hat[2],color='green',width=0.0005,zorder=1000)
		
		axarr[0].annotate(r'$\mathit{\vec{v}}_\mathregular{cm}$ = '+vcm_str+' km/s', 
			xy=(0.05, 0.8), xycoords='axes fraction', fontsize=10,color=txtclr)

		axarr[0].set_xlim(xl,xh)
		axarr[0].set_ylim(yl,yh)

		axarr[1].set_xlim(xl,xh)
		axarr[1].set_ylim(zl,zh)

		axarr[0].set_xlabel(r'$\mathit{x}$')
		axarr[0].set_ylabel(r'$\mathit{y}$')

		axarr[1].set_xlabel(r'$\mathit{x}$')
		axarr[1].set_ylabel(r'$\mathit{z}$')


	else:

		# circle1 = plt.Circle((0.83, -4.79), 0.3, ec='green',fc='none',lw=1,zorder=1e4)
		# axarr[0].add_artist(circle1)
		# circle2 = plt.Circle((0.83, -1.19), 0.3, ec='green',fc='none',lw=1,zorder=1e4)
		# axarr[1].add_artist(circle2)

		axarr[0].set_xlim(-bound,bound)
		axarr[0].set_ylim(-bound,bound)

		axarr[1].set_xlim(-bound,bound)
		axarr[1].set_ylim(-bound,bound)
		
	axarr[0].annotate(thislabel, xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12,color=txtclr)

	if show_progress:
		# header = snapHDF5.snapshot_header(fname)
		# time = np.round(header.time/0.7,1)
		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.round(np.array(tf['time'])[snapnum],2)
		tf.close()


		start_x = -4.5
		end_x = 4.5
		length = end_x - start_x
		frac = np.float(snapnum)/400.
		height = -4.5

		this_x = frac*length + start_x
		bar_x = np.array([start_x,this_x])
		bar_y = np.array([height,height])

		axarr[1].plot(bar_x,bar_y,'-',c='red',lw=3,alpha=0.4)
		axarr[1].annotate(r'$t$ = '+str(time)+' Gyr', xy=(start_x*0.98,height+0.2),fontsize=10,color='red',alpha=0.6)
		axarr[1].annotate('[',xy=(start_x-0.2,height-0.1),color='red',fontsize=10,alpha=0.4)
		axarr[1].annotate(']',xy=(end_x+0.1,height-0.1),color='red',fontsize=10,alpha=0.4)
	else:
		axarr[0].annotate('snapshot '+str(snapnum).zfill(3), xy=(0.2*bound, -0.85*bound), fontsize=10,color=txtclr)

	# figname = savedir#+'snapshot_'+str(snapnum).zfill(3)#+'.png'
	figname = '/home/ejahn003/movie_frames/'+sim+'/'

	# if do_rhalf:	figname += '_rhalf_'
	# if do_rcore:	figname += '_rcore_'
	
	figname += str(snapnum).zfill(3)+'.png'

	if save:
		print('saving figure: '+figname)
		plt.savefig(figname,format='png',dpi=200)
	else:
		print('showing figure: '+figname)
		plt.show()

def pltest():
	print('plotting test_rhoslope')
	fig,ax = p.makefig(1,figx=5,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	x = np.arange(0.1,0.21,0.01)
	a = 1e7

	# krange = np.array([-1,-0.5,-0.1])
	krange = np.linspace(-1,0,11)
	for k in krange:
		pl = a*(x**k)
		ax.plot(x,pl)
		a *= 1.5


	ax.set_xscale('log')
	ax.set_xlim(0.01,1)
	p.clear_axes(ax)
	# ax.set_xticks([0.1,0.2,0.5,1])
	# ax.set_xticklabels([0.1,0.2,0.5,1])	
	ax.set_xlabel(r'$r$ [kpc]')

	ax.set_yscale('log')
	ax.set_ylim(1e6,3e9)
	ax.set_ylabel(r'$\rho_\mathregular{dm}$ [M$_\odot$ kpc$^{-3}$]')

	p.finalize(fig,'test_rhoslope',save=save)

def power_radius_rcore():
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	nfw_dcut=10

	for i,sim in enumerate(models):
		f = h5py.File(d.datdir+'power_radius_'+sim+'.hdf5','r')
		power_radius = np.array(f['power_radius'])
		f.close()

		nf = h5py.File(d.datdir+'PITS_params_'+sim+'_dcut'+str(nfw_dcut)+'_hires.hdf5','r')
		core_radius = np.array(nf['rc'])
		nf.close()

		ax.plot(time,power_radius,ls='-',lw=1,c=colors_list[i])
		# ax.plot(time,core_radius,lw=1.3,c=colors_list[i],alpha=1)

	ax.axhline(y=0.556942176925,c='grey',ls=':',lw=1)

	ax.set_xlim(0,2.86)
	ax.set_xlabel('time [Gyr]')

	p.finalize(fig,'power_radius_'+whichsims,save=save)

def proj_panels(bound=5):
	print('plotting proj_panels')
	if not(whichsims == 'fixMorph_1e6'):
		raise ValueError('please choose whichsims = fixMorph_1e6')

	fig,axarr = p.makefig('10_proj')
	figname = '/home/ejahn003/plots/'+month+'/10panelproj_'+whichsims
		# models = models[1:]

	for j,sim in enumerate(models):

		# j = np.where(models==sim)[0][0] - 1

		# mypath = d.smuggledir+sim
		# allfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
		# snaplist = np.sort(allfiles[np.flatnonzero(np.core.defchararray.find(allfiles,'snapshot')!=-1)])
		# snapnum = int(snaplist[-1].split('.')[0][-3:])

		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		star_profile_all = np.array(f['type4'])
		f.close()

		# print(star_profile_all.shape)

		snapnum = star_profile_all.shape[0] - 1
		star_profile = star_profile_all[snapnum]
		
		print(sim+'   max snap = '+str(snapnum).zfill(3))
		fname = d.smuggledir+sim+'/snapshot_' + str(snapnum).zfill(3)

		plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
		# plt.style.use('dark_background')

		#---read-simulation-data-------------------------------------------------------------------
		gas_pos = snapHDF5.read_block(fname,"POS ",parttype=0)
		gasmass = snapHDF5.read_block(fname,"MASS", parttype=0)*(1.e10)/m.h
		str_pos = snapHDF5.read_block(fname,"POS ",parttype=4)
		strmass = snapHDF5.read_block(fname,"MASS", parttype=4)*(1.e10)/m.h
		darkpos = snapHDF5.read_block(fname,"POS ",parttype=1)
		darkmass = snapHDF5.read_block(fname,"MASS", parttype=1)*(1.e10)/m.h

		x_cm = np.sum(darkpos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(darkpos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(darkpos[:,2] * darkmass) / np.sum(darkmass)
		cm = np.array([x_cm,y_cm,z_cm]).T

		darkpos = darkpos - cm
		str_pos = str_pos - cm
		gas_pos = gas_pos - cm

		selstrbox = (np.abs(str_pos[:,0]) < bound) & (np.abs(str_pos[:,1]) < bound) & (np.abs(str_pos[:,2]) < bound)
		str_pos = str_pos[selstrbox]

		selgasbox = (np.abs(gas_pos[:,0]) < bound) & (np.abs(gas_pos[:,1]) < bound) & (np.abs(gas_pos[:,2]) < bound)
		gas_pos = gas_pos[selgasbox]

		#---projections----------------------------------------------------------------------------
		if sim=='fiducial_1e6' or sim=='rho0.1_1e6':
			gas_ap_face = 0.1
			gas_ap_edge = 0.1
			star_ap_face = 0.6
			star_ap_edge = 0.2
		elif sim=='illustris_1e6':
			gas_ap_face = 0.1
			gas_ap_edge = 0.02
			star_ap_face = 0.6
			star_ap_edge = 0.2
		elif sim=='tiny_1e6':
			gas_ap_face = 1
			gas_ap_edge = 0.7
			star_ap_face = 0.6
			star_ap_edge = 0.2
		elif sim=='compact_1e6':
			gas_ap_face = 0.2
			gas_ap_edge = 0.1
			star_ap_face = 0.6
			star_ap_edge = 0.2
		else:
			gas_ap_face = 0.3
			gas_ap_edge = 0.3
			star_ap_face = 0.2
			star_ap_edge = 0.2

		#---plot-face-on---
		sel_gas_front = (gas_pos[:,2] > 0)
		sel_gas_back = (gas_pos[:,2] <= 0)
		
		axarr[0,j].plot(gas_pos[:,0][sel_gas_front],gas_pos[:,1][sel_gas_front],',',c='blue',alpha=gas_ap_face,zorder=100)
		if np.sum(strmass) > 0:
			axarr[0,j].plot(str_pos[:,0],str_pos[:,1],',',c='darkorange',alpha=star_ap_face,zorder=10)
		axarr[0,j].plot(gas_pos[:,0][sel_gas_back],gas_pos[:,1][sel_gas_back],',',c='blue',alpha=gas_ap_face,zorder=1)	
		
		#---plot-edge-on---
		sel_gas_front = (gas_pos[:,1] > 0)
		sel_gas_back = (gas_pos[:,1] <= 0)	

		axarr[1,j].plot(gas_pos[:,0][sel_gas_front],gas_pos[:,2][sel_gas_front],',',c='blue',alpha=gas_ap_edge,zorder=100)	
		if np.sum(strmass) > 0:
			axarr[1,j].plot(str_pos[:,0],str_pos[:,2],',',c='darkorange',alpha=star_ap_edge,zorder=10)
		axarr[1,j].plot(gas_pos[:,0][sel_gas_back],gas_pos[:,2][sel_gas_back],',',c='blue',alpha=gas_ap_edge,zorder=1)

		#---plot-rhalf-----------------------------------------------------------------------------
		total_star_mass = np.sum(strmass)
		if total_star_mass > 0:
			# fractional_profile = star_profile / total_star_mass
			# ihalf = np.argmin(np.abs(fractional_profile-0.5))
			ihalf = m.find_nearest(star_profile,0.5*np.amax(star_profile),getindex=1)
			star_radius = drange[ihalf]
			print('half stellar mass radius: '+str(star_radius))
			circle1 = plt.Circle((0, 0), star_radius, ec='green',fc='none',lw=1,zorder=1e4)
			axarr[0,j].add_artist(circle1)
			circle2 = plt.Circle((0, 0), star_radius, ec='green',fc='none',lw=1,zorder=1e4)
			axarr[1,j].add_artist(circle2)

			if j==0:
				text = axarr[0,j].annotate(r'$r_{\mathregular{h}\ast}$', xy=(0.05, 0.89), xycoords='axes fraction', fontsize=10,color='green',zorder=1e4)
				text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='green', edgecolor='white'), path_effects.Normal()])

		#---plot-rcore-----------------------------------------------------------------------------
		nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut2.hdf5','r')
		core_radius = np.array(nf['core_radius'])[snapnum]
		nf.close()

		print('core radius: '+str(core_radius))
		circle3 = plt.Circle((0, 0), core_radius, ec='fuchsia',fc='none',lw=1,zorder=1e4)
		axarr[0,j].add_artist(circle3)
		circle4 = plt.Circle((0, 0), core_radius, ec='fuchsia',fc='none',lw=1,zorder=1e4)
		axarr[1,j].add_artist(circle4)

		if j==0:
			text = axarr[0,j].annotate(r'$r_\mathregular{core}$', xy=(0.05, 0.85), xycoords='axes fraction', fontsize=10,color='fuchsia',zorder=1e4)
			text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='fuchsia', edgecolor='white'), path_effects.Normal()])

		#---set-up-plot----------------------------------------------------------------------------
		axarr[0,j].set_xlim(-bound,bound)
		axarr[0,j].set_ylim(-bound,bound)

		axarr[1,j].set_xlim(-bound,bound)
		axarr[1,j].set_ylim(-bound,bound)

		text = axarr[0,j].annotate(models_label[j], xy=(0.05, 0.93), xycoords='axes fraction', fontsize=10,color='black',zorder=1e4)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])
		# text = axarr[0,j].annotate('snapshot '+str(snapnum).zfill(3), xy=(0.05, 0.89), xycoords='axes fraction', fontsize=8,color='black',zorder=1e4)
		# text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])

	#---plot-scale-length----------------------------------------------------------------------
	sl = bound/5
	ys = -0.9*bound
	xs = -0.9*bound
	axarr[0,0].plot(np.array([xs,xs+sl]),np.array([ys,ys]),c='red',lw=2,alpha=0.8,zorder=1e4)
	text = axarr[0,0].annotate(str(np.round(sl,1))+' kpc', xy=(-0.9*bound, -0.85*bound), fontsize=9,color='red',alpha=0.8,zorder=1e4)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='red', edgecolor='white'), path_effects.Normal()])


	#---save-figure-----
	print('saving figure: '+figname)
	plt.savefig(figname+'.png',format='png',dpi=200)

def rcore_compare():
	print('plotting rcore_compare ')

	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# print(ptypes)
	for i,sim in enumerate(models):
		print(sim)
		nfname = d.datdir+'coreradius_'+sim+'_dcut2_hires.hdf5'
		print(nfname)
		nf = h5py.File(nfname,'r')
		time = np.array(nf['time'])
		core_radius = np.array(nf['core_radius'])
		nfw_dcut = np.array(nf['nfw_dcut']).astype(float)
		ratio_cut = np.array(nf['ratio_cut']).astype(float)
		nf.close()
		ax.plot(time,core_radius,c=colors_list[i],lw=1,alpha=1)

		nfname = d.datdir+'coreradius_'+sim+'_dcut2.hdf5'
		print(nfname)
		nf = h5py.File(nfname,'r')
		time = np.array(nf['time'])
		core_radius = np.array(nf['core_radius'])
		nfw_dcut = np.array(nf['nfw_dcut']).astype(float)
		ratio_cut = np.array(nf['ratio_cut']).astype(float)
		nf.close()
		ax.plot(time,core_radius,c=colors_list[i],lw=3,alpha=0.5)

	ax.set_ylim(0,0.6)
	ax.set_yticks(np.arange(0,0.6,0.05),minor=True)
	ax.set_yticks(np.arange(0,0.7,0.1))
	ax.set_yticklabels([0,100,200,300,400,500,600])
	ax.set_ylabel(r'r$_\mathregular{core}$ [pc]')
	
	ax.legend(frameon=False,loc='upper left',prop={'size':11})
	ax.set_xlim([0,2.8])
	ax.set_xticks([0,0.5,1,1.5,2,2.5])
	ax.set_xticks([0.25,0.75,1.25,1.75,2.25,2.75],minor=True)
	ax.set_xlabel(r'time [Gyr]')

	p.finalize(fig,'rcore_compare',save=save,save_pdf=save_pdf,tight=True)

def rhalf(ptype):
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	ax.annotate(r'$M(r_\mathregular{50}) = 0.5 M(r=20\mathregular{kpc})$',xy=(0.25,0.9),xycoords='axes fraction',color='black',size=15)
	# ax.annotate(ptype,xy=(0.45,0.8),xycoords='axes fraction',color='black',size=15)

	for i,sim in enumerate(models):
		print(sim)
		text = ax.annotate(models_label[i],xy=(0.05,0.95-(0.06*i)),xycoords='axes fraction',fontsize=12,color=colors_list[i])#,alpha=alphas_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=colors_list[i], edgecolor='white'), path_effects.Normal()])
		try:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		except:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		
		drange = np.array(f['drange'])

		if ptype=='type23':
			type2_mass_all = np.array(f['type2'])
			type3_mass_all = np.array(f['type3'])
			max_snap = type3_mass_all.shape[0]
			time_plt = time[:max_snap]

			type2_rhalf_all = np.array([])
			type3_rhalf_all = np.array([])
			for snapnum in np.arange(0,max_snap):
				type2_mass_cur = type2_mass_all[snapnum]
				type3_mass_cur = type3_mass_all[snapnum]

				type2_mtotal = type2_mass_cur[m.find_nearest(drange,20,getindex=1)]
				type3_mtotal = type3_mass_cur[m.find_nearest(drange,20,getindex=1)]

				type2_ihalf = m.find_nearest(type2_mass_cur,0.5*type2_mtotal,getindex=1)
				type3_ihalf = m.find_nearest(type3_mass_cur,0.5*type3_mtotal,getindex=1)
				
				type2_rhalf_all = np.append(type2_rhalf_all, drange[type2_ihalf])
				type3_rhalf_all = np.append(type3_rhalf_all, drange[type3_ihalf])

			if i==0: lt2 = 'type2'; lt3 = 'type3'
			else:    lt2 = '';      lt3 = ''

			ax.plot(time_plt,type2_rhalf_all,c=colors_list[i],lw=1.6,ls='-', label=lt2)
			ax.plot(time_plt,type3_rhalf_all,c=colors_list[i],lw=1.6,ls='--',label=lt3)

		else:	
			mass_all = np.array(f[ptype])
			f.close()

			max_snap = mass_all.shape[0]
			time_plt = time[:max_snap]

			rhalf_all = np.array([])
			for snapnum in np.arange(0,max_snap):
				mass_cur = mass_all[snapnum]
				mtotal = mass_cur[m.find_nearest(drange,20,getindex=1)]
				ihalf = m.find_nearest(mass_cur,0.5*mtotal,getindex=1)
				rhalf_all = np.append(rhalf_all, drange[ihalf])
			ax.plot(time_plt,rhalf_all,c=colors_list[i],lw=1.6)
		
	if ptype=='type23': 	ax.legend(prop={'size':12},frameon=0)
	ax.set_xlabel('time [Gyr]')
	ax.set_xlim(0,np.amax(time))
	
	if ptype in np.array(['type2','type3','type23']): 
		ax.set_ylabel(r'$r_\mathregular{50 \ast}$ [kpc]',size=20)
		ax.set_ylim(0,3)
	elif ptype=='type4':
		ax.set_ylabel(r'$r_\mathregular{50 \ast}$ [kpc]',size=20)
		ax.set_ylim(0,8)
	elif ptype=='gas': 
		ax.set_ylabel(r'$r_\mathregular{50,gas}$ [kpc]',size=20)
		ax.set_ylim(0,20)
		ax.set_yticks(np.arange(21),minor=True)


	p.finalize(fig,'rhalf_'+ptype+'_'+whichsims,save=save,save_pdf=save_pdf)

def rhalf_sfgas(do_bins=True,do_weight=False):
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	coords = np.array([[0.1,0.95],
					  [0.1,0.9],
					  [0.1,0.85],
					  [0.25,0.95],
					  [0.25,0.9],
					  [0.25,0.85]])

	for i,sim in enumerate(models):
		print(sim)
		text = ax.annotate(models_label[i],xy=(coords[i,0],coords[i,1]),xycoords='axes fraction',fontsize=12,color=colors_list[i])#,alpha=alphas_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=colors_list[i], edgecolor='white'), path_effects.Normal()])
		
		if do_weight:
			f = h5py.File(d.datdir+'r50SFgas_massweight_'+sim+'.hdf5','r')
		else: 
			f = h5py.File(d.datdir+'r50SFgas_'+sim+'.hdf5','r')

		all_r50 = np.array(f['all_r50'])
		f.close()
		max_snap = len(all_r50)
		print(all_r50.shape)

		time_sim = time[:max_snap]

		fn = d.datdir+'num_sfr_time_'+sim+'txt'
		num_sfr = np.loadtxt(fn,dtype=float,delimiter=',')
		sel = num_sfr <= 10
		all_r50[sel] = np.nan

		if do_bins:
			dt = 0.05
			timebins = np.arange(0,np.amax(time_sim),dt)
			binwidth = (timebins[1] - timebins[0])/2
			r50_mean = np.array([])

			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				sel = (time > leftbin) & (time < rightbin) & np.logical_not(np.isnan(all_r50))
				r50_mean = np.append(r50_mean, np.mean(all_r50[sel]))

			ax.plot(timebins[0:-1]+binwidth,r50_mean,color=colors_list[i],lw=1.6)
			ax.plot(time_sim,all_r50,c=colors_list[i],lw=0.8,alpha=0.2)


		else:
			
			ax.plot(time,all_r50,c=colors_list[i],lw=1.6)


	ax.set_xlabel('time [Gyr]')
	ax.set_xlim(0,np.amax(time))

	ax.set_ylabel(r'$r_\mathregular{50,SFgas}$ [kpc]',size=20)

	if do_weight:
		ax.annotate('weighted by SF gas fraction',xy=(0.1,0.9),xycoords='axes fraction',color='black',size=12)
		# ax.set_ylim(0.1,10)
		ax.set_yscale('log')
		p.finalize(fig,'r50SFgas_weighted_'+whichsims,save=save)

	else:
		ax.set_ylim(0.1,10)
		ax.set_yscale('log')
		p.finalize(fig,'r50SFgas_'+whichsims,save=save)

def rho_hist(snapnum):
	# print('plotting rho histogram')
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	rho_bins = np.logspace(-6,6,100)

	for i,sim in enumerate(models):
		rhofile =  h5py.File(d.datdir+'rho_list_all_'+sim+'.hdf5','r')
		rho_from_file = np.array(rhofile['rho_1kpc_'+str(snapnum).zfill(3)])
		
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
		rho_from_snap = rho[d_gas < 1]

		ax.hist(rho_from_snap,histtype='step',color=colors_list[i],bins=rho_bins,alpha=alphas_list[i],lw=lw_list[i],ls='-')
		ax.hist(rho_from_file,histtype='step',color='blue',bins=rho_bins,alpha=alphas_list[i],lw=lw_list[i],ls='--')

		ax.annotate(models_label[i], xy=(0.05, 0.95-(i*0.05)), xycoords='axes fraction', fontsize=12,color=colors_list[i],alpha=alphas_list[i])

	#---------------------------------------------
	ax.annotate('snapshot '+str(snapnum).zfill(3), xy=(0.65, 0.95), xycoords='axes fraction', fontsize=12,color='k')
	ax.annotate('gas denisty within 1 kpc', xy=(0.52, 0.9), xycoords='axes fraction', fontsize=12,color='k')

	ax.set_xlabel(r'Density [cm$^\mathregular{-3}$]')
	ax.set_xscale('log')

	ax.set_ylabel('frequency')
	ax.set_yscale('log')

	# p.finalize(fig,'rho_hist_'+whichsims+'_'+str(snapnum).zfill(3),save=save)
	savedir = '/home/ejahn003/movie_frames/rho_histogram/'+str(snapnum).zfill(3)+'.png'
	print(savedir)
	plt.savefig(savedir,dpi=200,format='png')

def rho_hist_timeavg(do_median=True,normed=False):
	print('plotting rho histogram, time averaged')
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	scut = 10

	rho_bins = np.logspace(-6,6,100)
	bin_means = np.array([])

	for j in range(len(rho_bins)-1):
		this_bin_mean = (rho_bins[j] + rho_bins[j+1])/2
		bin_means = np.append(bin_means,this_bin_mean)

	for i,sim in enumerate(models):
		print(sim)
		f = h5py.File(d.datdir + 'rho_list_all_'+sim+'.hdf5','r')
		k = np.sort(np.array(f.keys()))
		max_snap = int(k[-1].split('_')[-1])


		for snapnum in np.arange(scut,max_snap+1):
			printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
			sys.stdout.write(printthing); sys.stdout.flush()
			if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
			else: print('')

			rho_list = np.array(f['rho_1kpc_'+str(snapnum).zfill(3)])
			if normed: hist, bin_edges = np.histogram(rho_list,bins=rho_bins,density=True)
			else:      hist, bin_edges = np.histogram(rho_list,bins=rho_bins)

			if snapnum==scut:	hist_all = hist
			else:				hist_all = np.vstack((hist_all,hist))

		f.close()

		hist_mean = np.array([])
		hist_stdv = np.array([])
		hist_stdv_p = np.array([])
		hist_stdv_n = np.array([])
		n_cols = hist_all.shape[1]
		for j in np.arange(n_cols):
			this_col = hist_all[:,j]
			this_col = this_col[np.logical_not(np.isnan(this_col))]

			hist_mean = np.append(hist_mean,np.median(this_col))
			hist_stdv_p = np.append(hist_stdv_p,np.std(this_col[this_col > np.median(this_col)]))
			hist_stdv_n = np.append(hist_stdv_n,np.std(this_col[this_col < np.median(this_col)]))
			fname = 'rho_hist_timeavg_median_'+whichsims
			ax.set_ylim(1e0,3e4)
			ax.set_xscale('log')
			ax.set_xlim(1e-3,2e4)
			ax.set_xticks([1e-3,1e-2,1e-1,1e0,1e1,1e2,1e3,1e4])
			
		ax.plot(bin_means,hist_mean,color=colors_list[i],alpha=alphas_list[i],lw=lw_list[i],zorder=1000/(i+1))

		hist_stdv_p[np.isnan(hist_stdv_p)] = 0
		hist_stdv_n[np.isnan(hist_stdv_n)] = 0

		if whichsims=='ff_ill_1e6':
			ax.fill_between(bin_means,hist_mean-hist_stdv_n,hist_mean+hist_stdv_p,color=colors_list[i],alpha=0.1,zorder=0)
			ax.plot(bin_means,hist_mean-hist_stdv_n,ls=':',color=colors_list[i],alpha=0.7,lw=0.5,zorder=1000/(i+1))
			ax.plot(bin_means,hist_mean+hist_stdv_p,ls='--',color=colors_list[i],alpha=0.7,lw=0.5,zorder=1000/(i+1))

		else:
			if not(sim in ['fiducial_1e6','illustris_1e6']):
				ax.fill_between(bin_means,hist_mean-hist_stdv_n,hist_mean+hist_stdv_p,color=colors_list[i],alpha=0.05,zorder=0)
				ax.plot(bin_means,hist_mean-hist_stdv_n,ls=':',color=colors_list[i],alpha=0.7,lw=0.5,zorder=1000/(i+1))
				ax.plot(bin_means,hist_mean+hist_stdv_p,ls='--',color=colors_list[i],alpha=0.7,lw=0.5,zorder=1000/(i+1))

		ax.annotate(models_label[i], xy=(0.05, 0.93-(i*0.05)), xycoords='axes fraction', fontsize=12,color=colors_list[i],alpha=alphas_list[i])
	
	#---------------------------------------------
	ax.annotate('time averaged gas denisty\nwithin 1 kpc', xy=(0.55, 0.88), xycoords='axes fraction', fontsize=12,color='k')

	ax.set_xlabel(r'$\rho_\mathregular{gas}$ [cm$^\mathregular{-3}$]')
	
	
	if normed:
		ax.set_ylabel('relative frequency')
	else:
		ax.set_ylabel('count')

	ax.set_yscale('log')
	# ax.set_yticks([])

	p.finalize(fig,fname,save=save,save_pdf=save_pdf)
	# savedir = '/home/ejahn003/movie_frames/rho_histogram/'+str(snapnum).zfill(3)+'.png'
	# print(savedir)
	# plt.savefig(savedir,dpi=200,format='png')

def rho_movie_frame(sim,snapnum,ptypespe):
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	if not(sim in models):
		raise ValueError('please choose sim in models')

	i = np.where(models==sim)[0][0]


	nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3'+'_hires.hdf5','r')
	time = np.array(nf['time'])[snapnum]
	core_radius = np.array(nf['core_radius'])[snapnum]
	nf.close()
	ax.axvline(x=core_radius,c='black',ls='--',lw=1.3,alpha=0.5,label=r'$r_\mathregular{core}$')

	f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
	drange = np.array(f['drange'])
	vols = 4./3.*np.pi*(drange**3.)

	rho_gas = np.array(f['gas'])[snapnum] / vols
	rho_dark = np.array(f['dark'])[snapnum] / vols
	rho_type2 = np.array(f['type2'])[snapnum] / vols
	rho_type3 = np.array(f['type3'])[snapnum] / vols
	rho_type4 = np.array(f['type4'])[snapnum] / vols
	
	f.close()

	rho_baryon = rho_gas + rho_type2 + rho_type3 + rho_type4

	if ptype=='each':
		fname = '/home/ejahn003/movie_frames/rhoprofile_'+sim+'/'+str(snapnum).zfill(3)+'.png'
		ax.plot(drange,rho_dark,c='black',lw=3,label='DM')
		ax.plot(drange,rho_baryon,c='darkorange',lw=2,label='baryons')
		ax.set_yscale('log')
		ax.set_ylabel(r'$\rho$ [M$_\odot$ kpc$^{-3}$]',size=20)
		ax.set_ylim(1e6,1e9)
	elif ptype=='frac':
		fname = '/home/ejahn003/movie_frames/rhofracprofile_'+sim+'/'+str(snapnum).zfill(3)+'.png'
		ax.plot(drange,rho_baryon/rho_dark,c='black',lw=2)
		ax.set_yscale('log')
		ax.set_ylabel(r'$\rho_\mathregular{bary} / \rho_\mathregular{DM}$',size=20)
		ax.set_ylim(1e-2,1e1)
		ax.axhline(y=1,c='black',alpha=0.3,ls=':',lw=0.9)

	ax.set_xscale('log')
	ax.set_xlim(0.1,10)
	ax.set_xticks([0.1,1,10])
	ax.set_xticklabels([0.1,1,10])
	ax.set_xlabel('dist [kpc]')

	ax.legend(prop={'size':12},frameon=False,loc='upper right')
	text = ax.annotate(models_label[i],xy=(0.05,0.9),xycoords='axes fraction',color='black',size=14)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=3, facecolor='black', edgecolor='white'), path_effects.Normal()])
	ax.annotate(r'$t$ = '+str(np.round(time,2))+' Gyr',xy=(0.45,0.9),xycoords='axes fraction',color='black',size=12)
	
	if save:
		print('saving: '+fname)
		plt.savefig(fname,format='png',dpi=200)
	else:
		print('showing: '+fname)
		plt.show()

def rho_sfr_movie_frame(sim,snapnum):
	fig,axarr = p.makefig('density movie')
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	try: i = np.where(models==sim)[0][0]
	except: raise ValueError('please choose whichsims that contains desired sim')

	#---plot-density-profile-and-core-radius-------------------------------------------------------
	f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	dark_mass_all = np.array(f['dark'])
	f.close()
	vols = 4./3.*np.pi*(drange**3)
	rho_init = dark_mass_all[0]/vols
	rho_this = dark_mass_all[snapnum]/vols #current snapshot

	# axarr[0].plot(drange,rho_init,'-',c='grey',lw=0.8)
	axarr[0].plot(drange,rho_this,'-',c=colors_list[i],lw=2)

	nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut2.hdf5','r')
	core_radius = np.array(nf['core_radius'])[snapnum]
	nf.close()

	axarr[0].axvline(x=core_radius,ls='--',c=colors_list[i],lw=1,alpha=0.5)
	axarr[0].annotate(models_label[i],xy=(0.7,0.9),xycoords='axes fraction',fontsize=12,color=colors_list[i],)#alpha=alphas_list[i])


	#---plot-sfr-and-current-time------------------------------------------------------------------
	filein = d.smuggledir + sim + '/sfr.txt'
	data = np.loadtxt(filein)
	time = data[:,0]/0.7
	sfr = data[:,2]
	sfr[(sfr == 0)]=1.e-10

	timebins = np.arange(0,2.86,0.025)
	binwidth = (timebins[1] - timebins[0])/2
	sfr_mean = np.array([])

	for j in range(len(timebins)-1):
		leftbin = timebins[j]
		rightbin = timebins[j+1]

		sel = (time > leftbin) & (time < rightbin)
		sfr_mean = np.append(sfr_mean, np.mean(sfr[sel]))

	axarr[1].plot(timebins[0:-1]+binwidth,sfr_mean,color='k',linewidth=1.8)

	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)	
	header = snapHDF5.snapshot_header(snapfile)
	current_time = header.time/0.7

	axarr[1].axvline(x=current_time,ls='-',c='k',lw=1)

	#---finish-plot--------------------------------------------------------------------------------

	axarr[0].set_xscale('log')
	axarr[0].set_xlim(0.1,10)
	axarr[0].set_xlabel('distance [kpc]')
	axarr[0].xaxis.set_label_position('top') 

	axarr[0].set_yscale('log')
	axarr[0].set_ylim(1e6,3e9)
	axarr[0].set_ylabel(r'$\rho_\mathregular{DM}$ [M$_\odot$ kpc$^{-3}$]')

	# axarr[1].set_xscale('log')
	axarr[1].set_xlim(0,2.86)
	axarr[1].set_xlabel('time [Gyr]')

	axarr[1].set_yscale('log')
	axarr[1].set_ylim(1e-3,5e0)
	axarr[1].set_ylabel(r'SFR [M$_\odot$ yr$^{-1}$]')

	# p.finalize(fig,'rho_sfr_movie_'+sim+'_'+str(snapnum).zfill(3),save=save)

	print('saving: '+str(snapnum).zfill(3)+'.png')

	# if save:
	plt.savefig('/home/ejahn003/movie_frames/rho_sfr_movie_'+sim+'/'+str(snapnum).zfill(3)+'.png',format='png',dpi=200)
	# else:
	# 	plt.show()

def sigma_all_timeavg():
	print('plotting sigma_radial_and_disk_timeavg')
	fig,axarr=p.makefig('3_vert',height=1,figx=7,figy=7)
	fig.subplots_adjust(hspace=0)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# axarr[1].axhline(y=1,ls=':',color='grey',lw=0.8)

	alphas_list[np.ones(len(models)).astype(bool)] = 1
	lw_list[np.ones(len(models)).astype(bool)] = 2
	drange = np.append(0.,np.logspace(-1,0.7,30))

	for i,sim in enumerate(models):
		print(sim)
		#--read-sigma-for-current-model----------------------------------------------
		sigmaname = d.datdir+'sigma_profiles_'+sim+'.hdf5'
		sigmafile = h5py.File(sigmaname,'r')
		k = np.array(sigmafile.keys())
		kl = np.sort(k[np.flatnonzero(np.core.defchararray.find(k,'sigma_star')!=-1)])
		max_snap = int(kl[-1].split('_')[-1])
		min_snap = 100

		for snapnum in np.arange(min_snap,max_snap+1):
			try:
				sigma_star = np.array(sigmafile['sigma_star_'+str(snapnum).zfill(3)])
				sigma_star_rho_this = sigma_star[:,0]
				sigma_star_phi_this = sigma_star[:,1]
				sigma_star_z_this = sigma_star[:,2]
			except:
				sigma_star_rho_this = np.zeros(30)
				sigma_star_phi_this = np.zeros(30)
				sigma_star_z_this = np.zeros(30)
			if snapnum==min_snap:
				sigma_star_rho_all = sigma_star_rho_this
				sigma_star_phi_all = sigma_star_phi_this
				sigma_star_z_all = sigma_star_z_this
			else:
				sigma_star_rho_all = np.vstack((sigma_star_rho_all,sigma_star_rho_this))
				sigma_star_phi_all = np.vstack((sigma_star_phi_all,sigma_star_phi_this))
				sigma_star_z_all = np.vstack((sigma_star_z_all,sigma_star_z_this))


		sigmafile.close()

		sigma_star_rho_avg = np.array([])
		sigma_star_phi_avg = np.array([])
		sigma_star_z_avg = np.array([])
		dplt = np.array([])

		# print('drange ',drange.shape)
		# print('sigma_star_rho_all ',sigma_star_rho_all.shape)

		for j in np.arange(len(drange)-1):
			sigma_star_rho_avg = np.append(sigma_star_rho_avg,np.median(sigma_star_rho_all[:,j]))
			sigma_star_phi_avg = np.append(sigma_star_phi_avg,np.median(sigma_star_phi_all[:,j]))
			sigma_star_z_avg = np.append(sigma_star_z_avg,np.median(sigma_star_z_all[:,j]))
			dplt = np.append(dplt,np.mean(np.array([drange[j],drange[j+1]])))

		axarr[0].plot(dplt,sigma_star_rho_avg,ls='-',c=colors_list[i],lw=lw_list[i],alpha=alphas_list[i])
		axarr[1].plot(dplt,sigma_star_phi_avg,ls='-',c=colors_list[i],lw=lw_list[i],alpha=alphas_list[i])
		axarr[2].plot(dplt,sigma_star_z_avg,ls='-',c=colors_list[i],lw=lw_list[i],alpha=alphas_list[i])

		simtext = models_label[i] #+ ' #' + str(snapnum).zfill(3)
		text = axarr[0].annotate(simtext,xy=(0.05,0.9-(i*0.08)),xycoords='axes fraction',fontsize=10,color=colors_list[i],alpha=alphas_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor=colors_list[i], 
			edgecolor='white'), path_effects.Normal()])

	axarr[0].set_ylim(0,60)
	axarr[0].set_yticks([0,10,20,30,40,50,60])
	axarr[0].set_yticks([5,15,25,35,45,55],minor=True)
	axarr[0].set_ylabel(r'$\mathregular{\sigma_r}$ [km s$\mathregular{^{-1}}$]')

	axarr[1].set_ylim(0,60)
	axarr[1].set_yticks([0,10,20,30,40,50,60])
	axarr[1].set_yticklabels(['0','10','20','30','40','50',''])
	axarr[1].set_yticks([5,15,25,35,45,55],minor=True)
	axarr[1].set_ylabel(r'$\mathregular{\sigma_\phi}$ [km s$\mathregular{^{-1}}$]')

	axarr[2].set_ylim(0,60)
	axarr[2].set_yticks([0,10,20,30,40,50,60])
	axarr[2].set_yticklabels(['0','10','20','30','40','50',''])
	axarr[2].set_yticks([5,15,25,35,45,55],minor=True)
	axarr[2].set_ylabel(r'$\mathregular{\sigma_z}$ [km s$\mathregular{^{-1}}$]')

	axarr[2].set_xlim(0.1,5)
	axarr[2].set_xscale('log')
	axarr[2].set_xlabel('d [kpc]')
	axarr[2].set_xticks([0.1,0.2,0.5,1.,2.,5.])
	axarr[2].set_xticklabels([0.1,0.2,0.5,1.,2.,5.])

	p.finalize(fig,'sigma_all_avg_'+whichsims,save=save,save_pdf=save_pdf)

def sfr_aperture(aperture,do_bins=0):
	print('plotting sfr_aperture')
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	for i,sim in enumerate(models):
		f = h5py.File(d.datdir+'SFRprofile_'+sim+'.hdf5','r')
		SFR_profile_all = np.array(f['SFR_profile_all'])
		drange = np.array(f['drange'])
		f.close()

		i_ap = m.find_nearest(drange,aperture,getindex=1)

		sfr_plt = np.array([])
		for snapnum in np.arange(0,SFR_profile_all.shape[0]):
			this_sfr_profile = SFR_profile_all[snapnum]
			sfr_plt = np.append(sfr_plt,this_sfr_profile[i_ap])

		ax.plot(time,sfr_plt,lw=1,c=colors_list[i])

	ax.set_xlim(0,2.86)
	ax.set_xlabel('time [Gyr]')

	ax.set_yscale('log')
	ax.set_ylabel(r'SFR (r < '+str(aperture)+')' + r' [M$_\odot$ yr$^{-1}$]')
	ax.set_ylim(1e-5,1e0)


	p.finalize(fig,'sfr_aperture_'+whichsims,save=save)

def sfr_masstime(do_bins=True,do_label=False):
	print('plotting sfr_masstime')
	fig,axarr=p.makefig('2_vert',height=1,figx=6,figy=7)
	fig.subplots_adjust(hspace=0)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for i,sim in enumerate(models):
		#---calculate-SFR--------------------------------------------------------------------------
		if do_label:
			text = axarr[0].annotate(models_label[i],xy=(0.95,0.93-(0.06*i)),xycoords='axes fraction',ha='right',fontsize=12,color=colors_list[i],alpha=alphas_list[i],zorder=1e6)
			text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=colors_list[i], 
				edgecolor='white'), path_effects.Normal()])

		sfrfile = h5py.File('sfr_time_'+sim+'.hdf5','r')
		sfr = np.array(sfrfile['sfr'])
		time = np.array(sfrfile['time'])
		sfrfile.close()

		#---plot-regular-sfr-----------------------------------------------------------------------
		# axarr[0].plot(time,sfr,color=colors_list[i],lw=lw_list[i]-0.5,alpha=alphas_list[i],zorder=1000/(i+1))

		#---bin-sfr--------------------------------------------------------------------------------
		if do_bins:
			dt = 0.025
			timebins = np.arange(0,2.86+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2
			sfr_mean = np.array([])

			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				sel = (time > leftbin) & (time < rightbin)
				sfr_mean = np.append(sfr_mean, np.mean(sfr[sel]))

			axarr[0].plot(timebins[0:-1]+binwidth,sfr_mean,color=colors_list[i],lw=lw_list[i],alpha=alphas_list[i])#,zorder=1000/(i+1)+10)
		else:
			axarr[0].plot(time,sfr,color=colors_list[i],lw=lw_list[i]-0.5,alpha=alphas_list[i])#,zorder=1000/(i+1)+10)


		#---calculate-mass-v-time------------------------------------------------------------------
		try: f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		except: f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		type4_profile = np.array(f['type4'])
		gas_profile = np.array(f['gas'])
		drange = np.array(f['drange'])
		f.close()

		i5 = m.find_nearest(drange,5,getindex=1)
		i1 = m.find_nearest(drange,1,getindex=1)
		type4mass = type4_profile[:,-1]
		gasmass_5 = gas_profile[:,i5]
		gasmass_1 = gas_profile[:,i1]

		if i==0: tmax = snapHDF5.snapshot_header(d.smuggledir + sim + '/snapshot_400').time / 0.7
		time = np.linspace(0,1,401) * tmax

		t90 = time[m.find_nearest(type4mass, 0.9*type4mass[-1], getindex=True)]
		t50 = time[m.find_nearest(type4mass, 0.5*type4mass[-1], getindex=True)]

		print(sim+'  final type 4 = '+m.scinote(type4mass[-1]))
		print(sim+'  t90 = '+str(np.round(t90,2))+' Gyr')
		print(sim+'  t50 = '+str(np.round(t50,2))+' Gyr')

		if sim=='rho0.1_1e6':
			axarr[1].plot(time[0:len(type4mass)],type4mass,lw=lw_list[i],c=colors_list[i],alpha=alphas_list[i],zorder=1000/(i+1),label=r'$M_\ast$')
			axarr[1].plot(time[0:len(gasmass_5)],gasmass_5,lw=lw_list[i],ls='--',c=colors_list[i],alpha=alphas_list[i],zorder=1000/(i+1),label=r'$M_\mathregular{gas}$ < 5 kpc')
			# axarr[1].plot(time[0:len(gasmass_1)],gasmass_1,lw=lw_list[i]-0.5,ls=':',c=colors_list[i],alpha=alphas_list[i],zorder=1000/(i+1),label=r'$M_\mathregular{gas}$ < 1 kpc')
		else:
			axarr[1].plot(time[0:len(type4mass)],type4mass,lw=lw_list[i],c=colors_list[i],alpha=alphas_list[i],zorder=1000/(i+1),)
			axarr[1].plot(time[0:len(gasmass_5)],gasmass_5,lw=lw_list[i],ls='--',c=colors_list[i],alpha=alphas_list[i],zorder=1000/(i+1))
			# axarr[1].plot(time[0:len(gasmass_1)],gasmass_1,lw=lw_list[i]-0.5,ls=':',c=colors_list[i],alpha=alphas_list[i],zorder=1000/(i+1))
		
	#axarr[1].annotate(r'solid: $M_\ast$',xy=(0.05,0.92),xycoords='axes fraction',fontsize=11,color='black')
	#axarr[1].annotate(r'dashed: $M_\mathregular{gas}$ (< 5kpc)',xy=(0.05,0.85),xycoords='axes fraction',fontsize=11,color='black')
	
	axarr[1].set_xlim([0,2.8])
	axarr[1].set_xticks([0,0.5,1,1.5,2,2.5])
	axarr[1].set_xticks([0.25,0.75,1.25,1.75,2.25,2.75],minor=True)
	axarr[1].set_xlabel(r'time [Gyr]')
	axarr[1].legend(prop={'size':11},frameon=False)

	axarr[0].set_yscale('log')
	axarr[0].set_ylim([1.e-3,1e1])
	# axarr[0].set_ylim([-0.01,0.5])
	axarr[0].set_ylabel(r'SFR [M$_\odot$/yr]')

	axarr[1].set_yscale('log')
	axarr[1].set_ylim(1e7,1.5*1.e9)
	axarr[1].set_ylabel(r'M [M$_\odot$]')
	
	p.finalize(fig,fname='sfr+masstime_'+whichsims,save=save,save_pdf=save_pdf,tight=True)

def sfr_profile_single(sim,snapnum):
	i = np.where(models==sim)[0][0]
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	ax.annotate(models_label[i],xy=(0.05,0.93),xycoords='axes fraction',color='black',size=12)

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])[snapnum]
	tf.close()

	ax.annotate(r'$t = $'+str(np.round(time,2))+' Gyr',xy=(0.05,0.86),xycoords='axes fraction',color='black',size=12)

	sfr_file = h5py.File(d.datdir+'SFRprofile_'+sim+'.hdf5','r')
	sfr_profile = np.array(sfr_file['SFR_profile_all'])[snapnum]
	drange = np.array(sfr_file['drange'])
	sfr_file.close()

	sel = (sfr_profile > 0)
	sfr_profile = sfr_profile[sel]
	drange = drange[sel]

	ax.plot(drange, sfr_profile,'-',c='black',lw=1.6)

	ax.set_xlabel(r'distance [kpc]')
	ax.set_xlim(0.01,5)
	ax.set_xticks([0.01,0.1,1,5])
	ax.set_xticklabels(['0.01','0.1','1','5'])
	ax.set_xscale('log')

	ax.set_yscale('log')
	ax.set_ylabel(r'SFR [M$_\odot$ yr$^{-1}$]')
	ax.set_ylim(1e-7,1e0)

	# p.finalize(fig,'r_sfr_'+sim,save=save)
	savename = '/home/ejahn003/movie_frames/sfr_profile_'+sim+'/'+str(snapnum).zfill(3)+'.png'
	print(savename)
	plt.savefig(savename,format='png',dpi=150)

def slopetime(do_bins=True):
	print('plotting slopetime')
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# if not(sim in models): raise ValueError('please choose sim in models')
	# i = np.where(models==sim)[0][0]
	# datdir = '/home/ethan/research/data/hdf5/'

	for i,sim in enumerate(models):
		print(sim)
		ax.annotate(models_label[i],xy=(0.75,0.2-(0.04*i)),xycoords='axes fraction',fontsize=11,color=colors_list[i],alpha=alphas_list[i])
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		vols = 4./3.*np.pi*(drange**3)
		dark_profile_all = np.array(f['dark'])
		f.close()

		max_snap = dark_profile_all.shape[0] 
		sel_fit = (drange > 0.1) & (drange < 0.3)
		drange_fit = drange[sel_fit]

		slope = np.array([])

		for snapnum in np.arange(0,max_snap):
			rho_dm = dark_profile_all[snapnum] / vols
			rho_dm_fit = rho_dm[sel_fit]
			fitted_pars = fit.powerlaw_sqfit(xdata=drange_fit,ydata=rho_dm_fit,a_guess=1e8,k_guess=-0.1)
			k = fitted_pars[1]
			slope = np.append(slope,k)

		time = np.arange(0,max_snap)/400. * 2.86

		ax.plot(time,slope,c=colors_list[i],lw=1,alpha=0.3,zorder=1000./(i+1.))

		dt = 0.05
		timebins = np.arange(0,np.amax(time)+dt,dt)
		binwidth = (timebins[1] - timebins[0])/2
		slope_mean = np.array([])

		for j in range(len(timebins)-1):
			leftbin = timebins[j]
			rightbin = timebins[j+1]

			sel = (time > leftbin) & (time < rightbin)
			slope_mean = np.append(slope_mean, np.mean(slope[sel]))

		ax.plot(timebins[0:-1]+binwidth,slope_mean,c=colors_list[i],lw=2,alpha=1,zorder=1000./(i+1.)+10)

		meanslope = np.mean(slope[(time > 0.75)])
		ax.axhline(y=meanslope,c=colors_list[i],lw=1,ls=':')


	ax.set_xlabel('time [Gyr]')
	ax.set_xlim(0,2.86)
	ax.set_xticks([0,0.5,1,1.5,2,2.5])
	ax.set_xticks([0.25,0.75,1.25,1.75,2.25,2.75],minor=True)

	ax.set_ylabel('power law slope')
	ax.set_ylim(-1,0.1)
	ax.set_yticks([-1.0,-0.8,-0.6,-0.4,-0.2,0])
	ax.set_yticks([-0.9,-0.7,-0.5,-0.3,-0.1],minor=True)


	p.finalize(fig,'powerlawslope_'+whichsims,save=save)

def test_eSFramp():
	print('plotting slopetime')
	fig,ax = p.makefig(1,figx=5,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	time = np.linspace(0,0.5,50) 

	'''
	if(All.Time < 0.1)
		current_SfrEfficiency = 0.01 + (All.SfrEfficiency - 0.01) * (All.Time / 0.1) ;
		else
		current_SfrEfficiency = All.SfrEfficiency;
    '''

	eSF = 1

	cur_eSF = np.array([])

	for t in time:
		if t<0.1: 
			this_eSF = 0.01 + (eSF - 0.01) * (t / 0.1)
			cur_eSF = np.append(cur_eSF, this_eSF)
		else:
			cur_eSF = np.append(cur_eSF, eSF)

	ax.plot(time,cur_eSF)

	ax.set_xlabel('time')
	ax.set_ylabel(r'$\varepsilon_\mathregular{sf}$')

	p.finalize(fig,'test_eSFramp',save=save)

def test_massprofiles(sim,snapnum=400,hires=True,density=True):
	fig,ax = p.makefig(1)
	if hires: f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
	else:     f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	gas_profile = np.array(f['gas'])[snapnum]
	dark_profile = np.array(f['dark'])[snapnum]
	type2_profile = np.array(f['type2'])[snapnum]
	type3_profile = np.array(f['type3'])[snapnum]
	type4_profile = np.array(f['type4'])[snapnum]
	f.close()

	if density:
		vols = 4./3.*np.pi*(drange**3)
		ax.plot(drange,gas_profile/vols,label='gas',lw=2)
		ax.plot(drange,dark_profile/vols,label='dark',lw=2)
		ax.plot(drange,type2_profile/vols,label='type 2',lw=2)
		ax.plot(drange,type3_profile/vols,label='type 3',lw=2)
		ax.plot(drange,type4_profile/vols,label='type 4',lw=2)
		ax.set_ylabel(r'$\rho_i$ [M$_\odot$ kpc$^{-3}$]')

	else:
		ax.plot(drange,gas_profile,label='gas',lw=2)
		ax.plot(drange,dark_profile,label='dark',lw=2)
		ax.plot(drange,type2_profile,label='type 2',lw=2)
		ax.plot(drange,type3_profile,label='type 3',lw=2)
		ax.plot(drange,type4_profile,label='type 4',lw=2)
		ax.set_ylabel(r'$M_i$ [M$_\odot$]')




	ax.legend(prop={'size':11},loc='lower right')

	ax.set_xlabel('distance [kpc]')
	ax.set_xscale('log')

	if hires: ax.set_xlim(0.01,100)
	else:     ax.set_xlim(0.1,250)

	
	ax.set_yscale('log')

	p.finalize(fig,'test_massprofiles',save=0)

def testproj(sim,snapnum,useCOMfile=True,bound=1,do_circles=0):
	if sim in models:
		i = np.where(models==sim)[0][0]
	else:
		raise ValueError('please choose a sim in models')

	fig, ax = p.makefig(n_panels=1,figx=5,figy=5)
	ax.tick_params(axis='both', which='both', top=False, bottom=False, labelbottom=False, labeltop=False,
			left=False, right=False, labelleft=False, labelright=False, direction='in',labelsize=11,length=5)


	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	# plt.style.use('dark_background')

	fname = d.smuggledir+sim+'/snapshot_' + str(snapnum).zfill(3)
	print(fname)

	darkpos = snapHDF5.read_block(fname,"POS ",parttype=1)/h
	darkmass = snapHDF5.read_block(fname,"MASS", parttype=1)*(1.e10)/m.h

	if useCOMfile:
		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()
		cm = CoM_all[snapnum]
	else:
		x_cm = np.sum(darkpos[:,0] * darkmass) / np.sum(darkmass)
		y_cm = np.sum(darkpos[:,1] * darkmass) / np.sum(darkmass)
		z_cm = np.sum(darkpos[:,2] * darkmass) / np.sum(darkmass)
		cm = np.array([x_cm,y_cm,z_cm]).T
	
	darkpos = darkpos - cm
	d_dark = np.linalg.norm(darkpos,axis=1)
	seldark_z_in_bound = (np.abs(darkpos[:,2]) < 0.02)
	darkpos = darkpos[seldark_z_in_bound]

	gas_pos = snapHDF5.read_block(fname,"POS ",parttype=0)/h - cm
	gasmass = snapHDF5.read_block(fname,"MASS", parttype=0)*(1.e10)/m.h
	d_gas = np.linalg.norm(gas_pos,axis=1)
	sel_gas_z_in_bound = (np.abs(gas_pos[:,2]) < 0.02)
	gas_pos = gas_pos[sel_gas_z_in_bound]

	type2_pos = snapHDF5.read_block(fname,"POS ",parttype=2)/h - cm
	type2mass = snapHDF5.read_block(fname,"MASS", parttype=2)*(1.e10)/m.h
	d_type2 = np.linalg.norm(type2_pos,axis=1)
	sel_type2_z_in_bound = (np.abs(type2_pos[:,2]) < 0.02)
	type2_pos = type2_pos[sel_type2_z_in_bound]
	
	type3_pos = snapHDF5.read_block(fname,"POS ",parttype=3)/h - cm
	type3mass = snapHDF5.read_block(fname,"MASS", parttype=3)*(1.e10)/m.h
	d_type3 = np.linalg.norm(type3_pos,axis=1)
	sel_type3_z_in_bound = (np.abs(type3_pos[:,2]) < 0.02)
	type3_pos = type3_pos[sel_type3_z_in_bound]


	# str_pos = snapHDF5.read_block(fname,"POS ",parttype=4)/h - cm
	# strmass = snapHDF5.read_block(fname,"MASS", parttype=4)*(1.e10)/m.h

	if do_circles:
		ax.plot(gas_pos[:,0],gas_pos[:,1],'o',ms=2,mew=0,mfc='blue',alpha=1,zorder=10)
		ax.plot(type2_pos[:,0],type2_pos[:,1],'o',ms=2,mew=0,mfc='orange',alpha=1,zorder=100)
		ax.plot(type3_pos[:,0],type3_pos[:,1],'o',ms=2,mew=0,mfc='red',alpha=1,zorder=1000)
		ax.plot(darkpos[:,0],darkpos[:,1],'o',ms=2,mew=0,mfc='black',alpha=1,zorder=0)
	else:
		ax.plot(gas_pos[:,0],gas_pos[:,1],',',c='blue',alpha=1,zorder=10)
		ax.plot(type2_pos[:,0],type2_pos[:,1],',',c='orange',alpha=1,zorder=100)
		ax.plot(type3_pos[:,0],type3_pos[:,1],',',c='red',alpha=1,zorder=1000)
	
	circle0 = plt.Circle((0, 0), 0.02, ec='black',fc='none',lw=1.5,zorder=1e4)
	ax.add_artist(circle0)

	ax.set_xlim(-bound,bound)
	ax.set_ylim(-bound,bound)

	text = ax.annotate(models_label[i], xy=(0.05, 0.93), xycoords='axes fraction', fontsize=12,color='black',zorder=1000)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])
	text = ax.annotate('snapshot '+str(snapnum), xy=(0.05, 0.87), xycoords='axes fraction', fontsize=12,color='black',zorder=1000)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])
	text = ax.annotate('bound = '+str(bound), xy=(0.05, 0.81), xycoords='axes fraction', fontsize=12,color='black',zorder=1000)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])

	n_dark_in_002 = np.count_nonzero(d_dark < 0.02)
	print('number of DM particles r < 0.02: '+str(n_dark_in_002))
	n_gas_in_002 = np.count_nonzero(d_gas < 0.02)
	print('number of gas particles r < 0.02: '+str(n_gas_in_002))
	n_type2_in_002 = np.count_nonzero(d_type2 < 0.02)
	print('number of type2 particles r < 0.02: '+str(n_type2_in_002))
	n_type3_in_002 = np.count_nonzero(d_type3 < 0.02)
	print('number of type3 particles r < 0.02: '+str(n_type3_in_002))


	plt.show()

def test_rhoslope(snapnum):
	print('plotting test_rhoslope')
	fig,ax = p.makefig(1,figx=5,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# datdir = '/home/ethan/research/data/hdf5/'
	for i,sim in enumerate(models):
		print(sim)
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		dark_profile_all = np.array(f['dark'])
		f.close()

		if snapnum > dark_profile_all.shape[0]:
			raise ValueError('please choose snapshot less than '+str(dark_profile_all.shape[0]))

		vols = 4./3.*np.pi*(drange**3)
		rho_dm = dark_profile_all[snapnum] / vols
		ax.plot(drange,rho_dm,'-',c=colors_list[i],lw=2.5)

		sel = (drange > 0.1) & (drange < 0.3)
		drange_fit = drange[sel]
		rho_dm_fit = rho_dm[sel]

		fitted_pars = fit.powerlaw_sqfit(xdata=drange_fit,ydata=rho_dm_fit,a_guess=1e8,k_guess=-0.1)
		powerlaw = fit.powerlaw_model(fitted_pars,drange_fit)

		# ax.plot(drange_fit,rho_dm_fit,'--',c='limegreen',lw=2,label='fitted region')
		ax.plot(drange_fit,powerlaw,'-.',c='lime',lw=2,label='power law fit')
		txt = models_label[i] + '; k = ' + str(np.round(fitted_pars[1],2))
		ax.annotate(txt,xy=(0.6,0.94-(0.04*i)),xycoords='axes fraction',fontsize=11,color=colors_list[i],alpha=alphas_list[i])

	ax.set_title('snapshot '+str(snapnum).zfill(3),size=10)

	ax.set_xscale('log')
	ax.set_xlim(0.1,5)
	p.clear_axes(ax)
	ax.set_xticks([0.1,1,5])
	ax.set_xticklabels([0.1,1,5])	
	ax.set_xlabel(r'$r$ [kpc]')

	ax.set_yscale('log')
	ax.set_ylim(1e7,1e9)
	ax.set_ylabel(r'$\rho_\mathregular{dm}$ [M$_\odot$ kpc$^{-3}$]')

	p.finalize(fig,'test_rhoslope',save=save)

def test_sfr_methods(sim):
	print('plotting test_sfr_methods')
	print(sim)
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	# plot SFR from direct output
	sfrfile = h5py.File('sfr_time_'+sim+'.hdf5','r')
	sfr = np.array(sfrfile['sfr'])
	time = np.array(sfrfile['time'])
	sfrfile.close()
	ax.plot(time,sfr,lw=1.5,c='orange',label='sfr.txt')

	# plot SFR from snapshots
	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	sfr_snap = np.array([])
	for snapnum in np.arange(0,401):
		printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(400).zfill(3)
		sys.stdout.write(printthing); sys.stdout.flush()
		if not(snapnum==400): sys.stdout.write("\b" * (len(printthing)))
		else: print('')
		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
		gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
		sfr_snap = np.append(sfr_snap, np.sum(gas_sfr))
	ax.plot(time,sfr_snap,lw=1.5,c='dodgerblue',label='snapshot')

	# finish figure
	ax.legend(prop={'size':12},loc='upper right',frameon=0)
	ax.annotate(sim,xy=(0.07,0.93),xycoords='axes fraction',size=12,color='black')

	ax.set_xlim(0,2.86)
	ax.set_xlabel('time [Gyr]')

	ax.set_yscale('log')
	ax.set_ylabel(r'SFR [M$_\odot$ yr$^{-1}$]')
	ax.set_ylim(1e-3,1e1)

	p.finalize(fig,'test_sfr_methods_'+sim,save=save)

def trace_particle_position(sim):
	print('plotting trace_particle_position')
	print(sim)
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	fname = d.datdir+'part_positions_'+sim+'.txt'
	all_data = np.loadtxt(fname,dtype=float,delimiter=',')
	trace_ids = all_data[0]
	dist_arr = all_data[1:]

	init_distances = np.array([0.1,0.2,0.5,1,2,5])
	colors = np.array(['black','blueviolet','dodgerblue','limegreen','orange','red'])

	# print(dist_arr.shape)
	# print(time.shape)

	fake = np.array([0,0,1,2,3,4])

	for i in np.arange(dist_arr.shape[1]):
		if not(i==1):
			ax.plot(time,dist_arr[:,i],'-',lw=1.6,c=colors[i],zorder=1000./(i+1))
			ax.annotate(str(init_distances[i]),xy=(0.63+(0.07*fake[i]),0.1),xycoords='axes fraction',color=colors[i],size=12)

	ax.annotate(sim,xy=(0.75,0.2),xycoords='axes fraction',color='black',size=12)
	ax.annotate(r'r$(t=0)$/kpc = ',xy=(0.42,0.1),xycoords='axes fraction',color='black',size=12)
	# ax.annotate('ID: '+str(trace_id),xy=(0.05,0.88),xycoords='axes fraction',color='black',size=12)

	ax.set_xlabel('time [Gyr]')
	ax.set_xlim(0,np.amax(time))

	ax.set_ylabel('distance [kpc]')
	ax.set_yscale('log')

	p.finalize(fig,'part_orbits_'+sim,save=save)

def twopanel_cores(do_bins=0,nfw_dcut=3,hires=True,doPITS=True,fixed_dist=True):
	fig,axarr = p.makefig('2_vert',height=1,figx=7,figy=8)
	fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	all_slopes = np.array([])

	for i,sim in enumerate(models):
		print(sim)

		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		#---plot-core-radius-on-top-frame----------------------------------------------------------
		if doPITS:
			if hires: 	nf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5','r')
			else: 		nf = h5py.File(d.datdir+'PITS_params_'+sim+'.hdf5','r')
			core_radius = np.array(nf['rc'])
			nf.close()
			
		else:
			# if hires:	nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'_hires.hdf5','r')
			# else: 		nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'.hdf5','r')
			try:    nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'_hires.hdf5','r')
			except: nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut'+str(nfw_dcut)+'.hdf5','r')
			core_radius = np.array(nf['core_radius'])
			nf.close()

		max_snap = core_radius.shape[0]
		time = time[:max_snap]

		if sim=='fiducial_1e6' and whichsims=='ff_ill_1e6':
			point_snaps = np.array([1,65,169,400]) 
			point_times = time[point_snaps]
			point_rcore = core_radius[point_snaps]
			axarr[0].plot(point_times,point_rcore,'s',ms=10,mew=1,mec='white',mfc='black')

		ag = np.amin(np.array([0.3,alphas_list[i]/2.]))

		#---smooth-rcore-by-binning----------------------------------------------------------------
		if do_bins:
			dt = 0.05
			timebins = np.arange(0,np.amax(time[:max_snap])+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2
			rcore_mean = np.array([])

			if i==3:
				print(timebins.shape)
				print(max_snap)

			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				sel = (time > leftbin) & (time < rightbin)
				rcore_mean = np.append(rcore_mean, np.mean(core_radius[sel]))
			time_bin = timebins[0:-1]+binwidth
			sel = (time_bin > 0.05)
			axarr[0].plot(time,core_radius,colors_list[i],lw=1,alpha=ag,zorder=1000./(i+1.)+10)
			axarr[0].plot(time_bin,rcore_mean,c=colors_list[i],lw=2,alpha=alphas_list[i],zorder=1000./(i+1.)+10)
		else:
			axarr[0].plot(time,core_radius,colors_list[i],lw=1.5,alpha=1,zorder=1000./(i+1.)+10)

		
		#---plot-power-law-slope-bottom-frame------------------------------------------------------
		# if doPITS and fixed_dist and hires: f = h5py.File(d.datdir+'PITSslope_'+sim+'_hires_fixed.hdf5','r')
		# if doPITS and hires and not(fixed_dist): f = h5py.File(d.datdir+'PITSslope_'+sim+'_hires.hdf5','r')
		# if doPITS and not(hires) and not(fixed_dist): f = h5py.File(d.datdir+'PITSslope_'+sim+'.hdf5','r')

		# if not(doPITS) and fixed_dist and hires: f = h5py.File(d.datdir+'powerlawslope_nfwdcut'+str(nfw_dcut)+'_'+sim+'_hires_fixed.hdf5','r')
		# if not(doPITS) and not(fixed_dist) and hires: f = h5py.File(d.datdir+'powerlawslope_nfwdcut'+str(nfw_dcut)+'_'+sim+'_hires.hdf5','r')
		# if not(doPITS) and not(fixed_dist) and not(hires): f = h5py.File(d.datdir+'powerlawslope_nfwdcut'+str(nfw_dcut)+'_'+sim+'.hdf5','r')
		
		try: f = h5py.File(d.datdir+'powerlawslope_nfwdcut'+str(nfw_dcut)+'_'+sim+'_hires.hdf5','r')
		except: f = h5py.File(d.datdir+'powerlawslope_nfwdcut'+str(nfw_dcut)+'_'+sim+'.hdf5','r')

		slope = np.array(f['slope'])
		time = np.array(f['time'])
		f.close()

		# tcut = 0.75
		tcut = 0

		all_slopes = np.append(all_slopes, slope[(time > tcut)])

		meanslope = np.mean(slope[(time > tcut)])
		axarr[1].axhline(y=meanslope,lw=1.5,ls='--',c=colors_list[i],alpha=alphas_list[i])

		sigma = np.std(slope[(time > tcut)])
		print('mean alpha = '+str(np.round(meanslope,5)))
		print('stdev alpha = '+str(np.round(sigma,5)))

		ag = np.amin(np.array([0.3,alphas_list[i]/2.]))
		axarr[1].plot(time[(time > 0.05)],slope[(time > 0.05)],c=colors_list[i],lw=1,alpha=ag,zorder=1000./(i+1.))

		#---smooth-slope-by-binning----------------------------------------------------------------
		dt = 0.05
		timebins = np.arange(0,np.amax(time)+dt,dt)
		binwidth = (timebins[1] - timebins[0])/2
		slope_mean = np.array([])

		for j in range(len(timebins)-1):
			leftbin = timebins[j]
			rightbin = timebins[j+1]

			sel = (time > leftbin) & (time < rightbin)
			slope_mean = np.append(slope_mean, np.mean(slope[sel]))
		time_bin = timebins[0:-1]+binwidth
		sel = (time_bin > 0.05)
		axarr[1].plot(time_bin[sel],slope_mean[sel],c=colors_list[i],lw=2,alpha=alphas_list[i],zorder=1000./(i+1.)+10)

		text = axarr[0].annotate(models_label[i],xy=(0.05,0.93-(0.06*i)),xycoords='axes fraction',fontsize=12,color=colors_list[i],alpha=alphas_list[i],zorder=1000./(i+1.)+30)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor=colors_list[i], 
			edgecolor='white'), path_effects.Normal()])
		print('')

	#--set-up-frames-------------------------------------------------------------------------------
	axarr[1].annotate(r'$\alpha_\mathregular{nfw} = -1$',xy=(0.1,0.1),xycoords='axes fraction',color='black',size=15)

	# print('mean slope (t>0.75) all = '+str(np.round(np.mean(all_slopes),5)))
	# print('standard deviation = '+str(np.round(np.std(all_slopes),5)))

	# ax.set_ylim(0,1)
	if doPITS:
		axarr[0].set_ylabel(r'r$_\mathregular{core}$ [kpc]')
		axarr[0].set_ylim(0,1.8)
		axarr[0].set_yticks(np.arange(0,1.9,0.1),minor=True)
		axarr[0].set_yticks(np.arange(0,1.8,0.5))
		text = axarr[0].annotate('pseudo isothermal sphere',xy=(0.35,0.88),xycoords='axes fraction',color='black',size=12,zorder=1e5)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black',edgecolor='white'), path_effects.Normal()])

		# axarr[0].set_yticklabels([0,100,200,300,400,500,600,700])
	else:
		axarr[0].set_ylabel(r'r$_\mathregular{core}$ [pc]')
		axarr[0].set_ylim(0,0.7)
		axarr[0].set_yticks(np.arange(0,0.8,0.05),minor=True)
		axarr[0].set_yticks(np.arange(0,0.8,0.1))
		axarr[0].set_yticklabels([0,100,200,300,400,500,600,700])

	axarr[0].legend(frameon=False,loc='upper left',prop={'size':11})
	
	axarr[1].set_xlim([0,2.86])
	axarr[1].set_xticks([0,0.5,1,1.5,2,2.5])
	axarr[1].set_xticks([0.25,0.75,1.25,1.75,2.25,2.75],minor=True)
	axarr[1].set_xlabel(r'time [Gyr]')

	axarr[1].set_ylabel(r'$\alpha$',size=20)
	axarr[1].set_ylim(-0.8,0.1)
	axarr[1].set_yticks([-0.8,-0.6,-0.4,-0.2,0])
	axarr[1].set_yticks([-0.7,-0.5,-0.3,-0.1],minor=True)

	fname = 'rcore_slope_'

	if doPITS: fname += 'PITS_'
	if fixed_dist: fname += 'fixdist_'
	if hires: fname += 'hires_'

	fname += whichsims

	p.finalize(fig,fname,save=save,save_pdf=save_pdf,tight=True)

def vcirc_sfr_movie_frame(sim,snapnum):
	fig,axarr = p.makefig('density movie')
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	try: i = np.where(models==sim)[0][0]
	except: raise ValueError('please choose whichsims that contains desired sim')

	#---plot-density-profile-and-core-radius-------------------------------------------------------
	axarr[0].annotate(models_label[i],xy=(0.7,0.9),xycoords='axes fraction',fontsize=12,color=colors_list[i],)#alpha=alphas_list[i])

	f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
	drange = np.array(f['drange'])
	gas_profile = np.array(f['gas'])[snapnum]
	dark_profile = np.array(f['dark'])[snapnum]
	type2_profile = np.array(f['type2'])[snapnum]
	type3_profile = np.array(f['type3'])[snapnum]
	type4_profile = np.array(f['type4'])[snapnum]
	f.close()
	vols = 4./3.*np.pi*(drange**3)

	nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut2.hdf5','r')
	core_radius = np.array(nf['core_radius'])[snapnum]
	nf.close()
	axarr[0].axvline(x=core_radius,ls='--',c=colors_list[i],lw=1,alpha=0.5)

	mstr = type4_profile[-1]
	ihalf = m.find_nearest(type4_profile[(drange < 10)],0.5*mstr,getindex=True)
	rhalf = drange[(drange < 10)][ihalf]
	axarr[0].axvline(x=rhalf,ls='-',c='orange',lw=1,alpha=0.8)

	total_profile = gas_profile + dark_profile + type2_profile + type3_profile + type4_profile
	vcirc_total = np.sqrt(m.Gprime*total_profile/drange)
	axarr[0].plot(drange,vcirc_total,'-',c=colors_list[i],lw=2)

	vcirc_dm = np.sqrt(m.Gprime*dark_profile/drange)
	axarr[0].plot(drange,vcirc_dm,'--',c=colors_list[i],lw=1.5)

	#---plot-sfr-and-current-time------------------------------------------------------------------
	filein = d.smuggledir + sim + '/sfr.txt'
	data = np.loadtxt(filein)
	time = data[:,0]/0.7
	sfr = data[:,2]
	sfr[(sfr == 0)]=1.e-10

	timebins = np.arange(0,2.86,0.025)
	binwidth = (timebins[1] - timebins[0])/2
	sfr_mean = np.array([])

	for j in range(len(timebins)-1):
		leftbin = timebins[j]
		rightbin = timebins[j+1]

		sel = (time > leftbin) & (time < rightbin)
		sfr_mean = np.append(sfr_mean, np.mean(sfr[sel]))

	axarr[1].plot(timebins[0:-1]+binwidth,sfr_mean,color='k',linewidth=1.8)

	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)	
	header = snapHDF5.snapshot_header(snapfile)
	current_time = header.time/0.7

	axarr[1].axvline(x=current_time,ls='-',c='k',lw=1)

	#---finish-plot--------------------------------------------------------------------------------

	axarr[0].set_xlim(0,5)
	axarr[0].set_xlabel('distance [kpc]')
	axarr[0].xaxis.set_label_position('top') 

	axarr[0].set_ylim(0,90)
	axarr[0].set_ylabel(r'$v_\mathregular{circ}$ [km s$^{-1}$]')

	# axarr[1].set_xscale('log')
	axarr[1].set_xlim(0,2.86)
	axarr[1].set_xlabel('time [Gyr]')

	axarr[1].set_yscale('log')
	axarr[1].set_ylim(1e-3,5e0)
	axarr[1].set_ylabel(r'SFR [M$_\odot$ yr$^{-1}$]')

	# p.finalize(fig,'rho_sfr_movie_'+sim+'_'+str(snapnum).zfill(3),save=save)

	print('saving: '+str(snapnum).zfill(3)+'.png')

	plt.savefig('/home/ejahn003/movie_frames/vcric_sfr_movie_'+sim+'/'+str(snapnum).zfill(3)+'.png',format='png',dpi=200)
	# plt.show()

def virial_radius():
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	for i,sim in enumerate(models):
		f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		vols = 4./3. * np.pi * (drange**3.)
		rho_DM_all = np.array(f['dark']) / vols
		f.close()
		
		r200_all = np.array([])
		for snapnum in np.arange(401):
			rho_DM = rho_DM_all[snapnum]
			H = 70./1000.
			rho_crit = 3*H**2 / (8*np.pi*m.Gprime) 
			r200 = drange[(rho_DM >= 200*rho_crit)][-1]
			r200_all = np.append(r200_all,r200)

		if i==0:
			ax.plot(time,r200_all,lw=3,c=colors_list[i])
		else:
			ax.plot(time,r200_all,lw=1,c=colors_list[i])

		print(sim,np.mean(r200_all))
	ax.set_xlim(0,2.86)
	ax.set_xlabel('time [Gyr]')
	ax.set_yscale('log')
	ax.set_ylim(1,1000)
	
	p.finalize(fig,'r200_'+whichsims,save=save)

def vphi_profile(do_cold=False):
	print('plotting vphi_profile')
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	for i,sim in enumerate(models):
		mypath = d.smuggledir+sim
		allfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
		snaplist = np.sort(allfiles[np.flatnonzero(np.core.defchararray.find(allfiles,'snapshot')!=-1)])
		snapnum = int(snaplist[-1].split('.')[0][-3:])
		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
		print(sim+'  -  '+str(snapnum).zfill(3))

		# print(snapfile)
		# gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
		gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h
		gas_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=0)

		# strmass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
		str_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h
		str_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=4)

		# darkmass = snapHDF5.read_block(snapfile, 'MASS', parttype=1)*(1.e10)/h
		# dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h
		# dark_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=1)

		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		dark_cm = np.array(cm_file['CoM'])[snapnum]
		cm_file.close()

		v_cm_file = h5py.File(d.datdir+'centerofmassvelocity_'+sim+'.hdf5','r')
		dark_v_cm = np.array(v_cm_file['Vel_CoM'])[snapnum]
		v_cm_file.close()

		xy_cut = 5
		z_cut = 0.5

		#convert gas to cylindrical coordinates ---------------------------------------------------
		if do_cold:
			rho = snapHDF5.read_block(snapfile,"RHO ",parttype=0)
			rho *= m.Xh/m.PROTONMASS*m.UnitDensity_in_cgs  #now in cm^{-3} h^3 
			U = snapHDF5.read_block(snapfile,"U   ",parttype=0)
			Nelec = snapHDF5.read_block(snapfile,"NE  ",parttype=0)
			MeanWeight= 4.0/(3*m.Xh+1+4*m.Xh*Nelec) * m.PROTONMASS
			temp = MeanWeight/m.BOLTZMANN * (m.gamma-1) * U * m.UnitEnergy_in_cgs/ m.UnitMass_in_g
			sel_cold = temp < 1e4
			gas_pos = gas_pos[sel_cold]
			gas_vel = gas_vel[sel_cold]

		gas_pos = gas_pos-dark_cm
		gas_vel = gas_vel-dark_v_cm
		
		r_gas = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2)
		d_gas = np.linalg.norm(gas_pos-dark_cm, axis=1)

		rho_gas = np.sqrt(gas_pos[:,0]**2 + gas_pos[:,1]**2)
		phi_gas = np.arctan(gas_pos[:,0]/gas_pos[:,1])

		vphi_gas = (gas_pos[:,0]*gas_vel[:,1] - gas_pos[:,1]*gas_vel[:,0])*(np.cos(phi_gas) - np.sin(phi_gas))/rho_gas

		sel_box = (np.abs(gas_pos[:,0]) < xy_cut) & (np.abs(gas_pos[:,1]) < xy_cut) & (np.abs(gas_pos[:,2]) < z_cut)
		vphi_gas = vphi_gas[sel_box]
		r_gas = r_gas[sel_box]

		#convert stars to cylindrical coordinates -------------------------------------------------
		str_pos = str_pos-dark_cm
		str_vel = str_vel-dark_v_cm

		r_str = np.sqrt(str_pos[:,0]**2 + str_pos[:,1]**2)
		d_str = np.linalg.norm(str_pos-dark_cm, axis=1)

		rho_str = np.sqrt(str_pos[:,0]**2 + str_pos[:,1]**2)
		phi_str = np.arctan(str_pos[:,0]/str_pos[:,1])

		vphi_str = (str_pos[:,0]*str_vel[:,1] - str_pos[:,1]*str_vel[:,0])*(np.cos(phi_str) - np.sin(phi_str))/rho_str

		sel_box = (np.abs(str_pos[:,0]) < xy_cut) & (np.abs(str_pos[:,1]) < xy_cut) & (np.abs(str_pos[:,2]) < z_cut)
		vphi_str = vphi_str[sel_box]
		r_str = r_str[sel_box]

		#calculate medians ------------------------------------------------------------------------
		# drange = np.logspace(-1,1,20)
		drange = np.linspace(0,5,10)
		vphi_gas_median = np.array([])
		r_gas_median = np.array([])
		vphi_gas_err = np.array([])

		vphi_str_median = np.array([])
		r_str_median = np.array([])
		vphi_str_err = np.array([])

		for j in range(len(drange)-1):
			sel = (r_gas > drange[j]) & (r_gas < drange[j+1])
			# this_vphi = vphi_gas[sel]
			vphi_gas_median = np.append(vphi_gas_median,np.median(vphi_gas[sel]))
			vphi_gas_err = np.append(vphi_gas_err,(np.std(vphi_gas[sel])/np.sqrt(np.count_nonzero(sel))))
			r_gas_median = np.append(r_gas_median,np.median(r_gas[sel]))

			sel = (r_str > drange[j]) & (r_str < drange[j+1])
			# this_vphi = vphi_str[sel]
			vphi_str_median = np.append(vphi_str_median,np.median(vphi_str[sel]))
			vphi_str_err = np.append(vphi_str_err,(np.std(vphi_str[sel])/np.sqrt(np.count_nonzero(sel))))
			r_str_median = np.append(r_str_median,np.median(r_str[sel]))

		ax.plot(r_gas_median,vphi_gas_median,ls='-',marker='s',c=colors_list[i],mew=0,ms=7,lw=lw_list[i],zorder=1000/(i+1),alpha=0.5)
		# ax.plot(r_gas_median,vphi_gas_median,'s',mfc=colors_list[i],mew=0,ms=7,zorder=1000/(i+1),alpha=0.5,lw=0)
		# ax.errorbar(r_gas_median,vphi_gas_median,yerr=vphi_gas_err,c=colors_list[i],elinewidth=0.8,capsize=5,zorder=1000/(i+1),alpha=0.5)

		ax.plot(r_str_median,vphi_str_median,ls=':',marker='^',mew=0,ms=7,c=colors_list[i],lw=lw_list[i],zorder=1000/(i+1),alpha=1)
		# ax.plot(r_str_median,vphi_str_median,'^',mfc=colors_list[i],mew=0,ms=7,zorder=1000/(i+1),alpha=1,lw=0)
		# ax.errorbar(r_str_median,vphi_str_median,yerr=vphi_str_err,c=colors_list[i],elinewidth=0.8,capsize=5,zorder=1000/(i+1),alpha=1)

		# ax.plot(r_gas,vphi,'o',mew=0,mfc=colors_list[i],ms=1.5,alpha=0.05,zorder=i*10)

		# if snapnum==400: mt = models_label[i]
		# else: mt = models_label[i]+'('+str(snapnum).zfill(3)+')'

		text = ax.annotate(models_label[i],xy=(0.05,0.95-(0.05*i)),
			xycoords='axes fraction',fontsize=11,color=colors_list[i],zorder=i*10+1000,alpha=alphas_list[i])
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor=colors_list[i], 
			edgecolor='white'), path_effects.Normal()])

		#---plot-vcirc-----------------------------------------------------------------------------
		# if not(sim=='ff_tiny_1e6'):
		try: f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		except: f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')
		drange = np.array(f['drange'])
		gas_profile = np.array(f['gas'])[snapnum]
		dark_profile = np.array(f['dark'])[snapnum]
		type2_profile = np.array(f['type2'])[snapnum]
		type3_profile = np.array(f['type3'])[snapnum]
		type4_profile = np.array(f['type4'])[snapnum]
		f.close()

		total_profile = gas_profile + dark_profile + type2_profile + type3_profile + type4_profile
		ax.plot(drange,np.sqrt(m.Gprime*dark_profile/drange),c=colors_list[i],ls='--',lw=lw_list[i],zorder=10/(i+1),alpha=alphas_list[i])

	#---plotting-stuff-----------------------------------------------------------------------------
	# ax.annotate('gas rotation',xy=(0.8,0.95),xycoords='axes fraction',fontsize=11,color='k')
	# ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.8,0.9),xycoords='axes fraction',fontsize=11,color='k')
	ax.annotate(r'---  $v_\mathregular{circ}$',xy=(0.84,0.84),xycoords='axes fraction',fontsize=14,color='k')

	if do_cold: 
		ax.plot(1e12,1e12,ls='-',marker='s',c='black',mew=0,ms=7,lw=lw_list[i],alpha=0.5,label='cold gas')
		fname = 'vphi_coldgas_'+whichsims
	else:       
		ax.plot(1e12,1e12,ls='-',marker='s',c='black',mew=0,ms=7,lw=lw_list[i],alpha=0.5,label='gas')
		fname = 'vphi_'+whichsims
	

	ax.plot(1e12,1e12,ls=':',marker='^',mew=0,ms=7,c='black',lw=lw_list[i],alpha=1,label='stars')
	# ax.plot(1e12,1e12,ls='--',c='black',lw=lw_list[i],alpha=1,label=r'$v_\mathregular{circ}$ (DM)')

	ax.legend(loc='upper right',frameon=False,prop={'size':10})

	ax.set_xlim(0,5)
	ax.set_xticks([0,1,2,3,4,5])
	ax.set_xticks(np.arange(0,5,0.5),minor=True)
	ax.set_xlabel(r'$r$ [kpc]')

	ax.set_ylim(0,90)
	# ax.set_yscale('log')
	ax.set_ylabel(r'$v_\phi$ [km s$^{-1}$]')

	p.finalize(fig,fname,save=save,save_pdf=save_pdf)



#---new-ones---------------------------------------------------------------------------------------
def test_age(snapnum):
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	text = ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.75,0.15),xycoords='axes fraction',color='black',size=12)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])

	text = ax.annotate(r'$t$ = '+str(np.round(time[snapnum],2)).zfill(3)+' Gyr',xy=(0.75,0.1),xycoords='axes fraction',color='black',size=12)
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])

	cur_time = time[snapnum]

	for i,sim in enumerate(models):
		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
		star_age = (cur_time - snapHDF5.read_block(snapfile, 'AGE ', parttype=4)/h )* 1.e3

		ax.hist(star_age,bins=1000,histtype='step',color=colors_list[i],normed=False,lw=1.5,log=True)
		ax.annotate(models_label[i],xy=(0.05,0.95-(0.05*i)),xycoords='axes fraction',color=colors_list[i],size=12)
	
	ax.set_xlabel('star age [Myr]')
	ax.set_xticks(np.arange(0,105,5),minor=True)
	ax.set_xlim(0,100)

	ax.set_ylabel('count')
	ax.set_ylim(1e-1,1e3)

	if save:
		sd = '/home/ejahn003/movie_frames/star_age_'+whichsims+'/' + str(snapnum).zfill(3)+'.png'
		print('saving fig: '+sd)
		plt.savefig(sd,format='png',dpi=150)
	else:
		plt.show()

def r50young(age_cut=50,do_median=True,do_bins=False):
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	ax.annotate('star age < '+str(age_cut)+' Myr',xy=(0.05,0.9),xycoords='axes fraction',color='black',size=12)

	for i,sim in enumerate(models):
		ax.annotate(models_label[i],xy=(0.75,0.95-(i*0.05)),xycoords='axes fraction',color=colors_list[i],size=12)

		if do_median:
			rf = h5py.File(d.datdir+'rmedianyoung_'+str(age_cut)+'_'+sim+'.hdf5')
			r50 = np.array(rf['all_rmedian'])
			rf.close()
			ax.set_ylabel(r'$<r_\mathregular{young}>$ [kpc]',size=20)
		else:
			rf = h5py.File(d.datdir+'r50young_'+str(age_cut)+'_'+sim+'.hdf5')
			r50 = np.array(rf['all_r50'])
			rf.close()
			ax.set_ylabel(r'$r_\mathregular{50, young}$ [kpc]',size=20)

		selpos = r50 > 0

		if do_bins:
			ax.plot(time[selpos],r50[selpos],c=colors_list[i],lw=0.8,alpha=0.3)
			dt = 0.05
			timebins = np.arange(0,np.amax(time)+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2

			r50_mean = np.array([])
			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				selbin = (time[selpos] > leftbin) & (time[selpos] < rightbin)
				r50_mean = np.append(r50_mean, np.mean(r50[selpos][selbin]))

			time_plt = timebins[0:-1]+binwidth
			ax.plot(time_plt,r50_mean,color=colors_list[i],lw=lw_list[i],alpha=alphas_list[i],ls='-',zorder=10*(i+1))

		else:
			ax.plot(time[selpos],r50[selpos],c=colors_list[i],lw=1.3)
		# ax.hist(r50,bins=100,histtype='step',color=colors_list[i],log=False,lw=1.5)

	ax.set_xlim(0,2.86)
	ax.set_xlabel('time [Gyr]')

	ax.set_yscale('log')
	ax.set_ylim(1e-1,1e2)

	p.finalize(fig,'r50young_'+str(age_cut)+'Myr_'+whichsims,save=save)

def youngstar_dhist(age_cut,snapnum):
	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	
	ax.annotate(r'$t$ = '+str(np.round(time[snapnum],2))+' Gyr',xy=(0.05,0.9),xycoords='axes fraction',color='black',size=12)
	ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.05,0.85),xycoords='axes fraction',color='black',size=12)
	ax.annotate('star age < '+str(age_cut)+' Myr',xy=(0.05,0.8),xycoords='axes fraction',color='black',size=12)

	for i,sim in enumerate(models):
		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()

		snapfile = d.smuggledir + sim + '/snapshot_' + str(snapnum).zfill(3)
		star_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h - CoM_all[snapnum]
		star_age = (time[snapnum] - (snapHDF5.read_block(snapfile, 'AGE ', parttype=4)/h)) * 1.e3

		sel = (star_age < age_cut)

		try:
			d_young = np.linalg.norm(star_pos[sel],axis=1)

			ax.hist(d_young,bins=np.logspace(-1,1,101),histtype='step',color=colors_list[i],normed=False,lw=1.5,log=True)
			ax.annotate(models_label[i]+r'; $n$ = '+str(len(d_young)),xy=(0.65,0.9-(i*0.05)),xycoords='axes fraction',color=colors_list[i],size=12)
		except:
			ax.annotate(models_label[i]+r'; $n$ = 0',xy=(0.65,0.9-(i*0.05)),xycoords='axes fraction',color=colors_list[i],size=12)

		# ax.annotate(r'$n$ = '+str(len(d_young)),xy=(0.05,0.8-(i*0.05)),xycoords='axes fraction',color=colors_list[i],size=12)

	ax.set_xlabel('distance [kpc]')
	ax.set_xscale('log')
	ax.set_xticks([0.1,1,10])
	ax.set_xticklabels(['0.1','1','10'])
	ax.set_xlim(0.1,10)

	ax.set_ylabel('count')
	ax.set_ylim(1e-1,1e3)

	if save:
		sd = '/home/ejahn003/movie_frames/youngstar_dhist_'+whichsims+'/' + str(snapnum).zfill(3)+'.png'
		print('saving fig: '+sd)
		plt.savefig(sd,format='png',dpi=150)
	else:
		plt.show()

def mstar_radial(do_bins=True):
	fig,axarr = p.makefig('4_vert',figx=6.67,figy=10)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	# axarr[0].annotate('  cumulative\nfractional SFH',xy=(0.6,0.4),xycoords='axes fraction',size=11,color='black')

	for i,sim in enumerate(models):
		print(sim)
		# axarr[0].annotate(models_label[i],xy=(0.05,0.88-(i*0.1)),xycoords='axes fraction',size=12,color=colors_list[i])

		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()

		mypath = d.smuggledir+sim
		onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
		a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
		a = np.sort(a)
		max_snap = int(a[-1].split('.')[0][-3:])

		#---calculate-mass-within-different-bins-------------------------------------------------------
		d_bins = np.array([0,0.2,0.5,1,np.inf])
		fname = d.datdir+'mstr_bins_'+sim+'.txt'

		try:
			all_m_bins = np.loadtxt(fname,dtype=float,delimiter=',')
			print('successfully read data from file')

		except:
			for snapnum in np.arange(0,max_snap+1):
				printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
				sys.stdout.write(printthing); sys.stdout.flush()
				if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
				else: print('')
				
				snapfile = mypath+'/snapshot_'+str(snapnum).zfill(3)
				star_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h - CoM_all[snapnum]
				starmass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
				star_age = (time[snapnum] - (snapHDF5.read_block(snapfile, 'AGE ', parttype=4)/h)) * 1.e3 # now in Myr

				sel_young = star_age < 5

				if np.count_nonzero(sel_young) > 10:
					star_pos = star_pos[sel_young]
					starmass = starmass[sel_young]

					d_star = np.linalg.norm(star_pos,axis=1)

					this_m_bins = np.array([])
					for j in np.arange(len(d_bins)-1):
						d_in  = d_bins[j]
						d_out = d_bins[j+1]

						sel = (d_star >= d_in) & (d_star < d_out)
						this_m_bins = np.append(this_m_bins, np.sum(starmass[sel]))
				else:
					this_m_bins = np.zeros(len(d_bins)-1)

				if snapnum == 0:
					all_m_bins = this_m_bins
				else:
					all_m_bins = np.vstack((all_m_bins,this_m_bins)) 

			np.savetxt(fname,all_m_bins,delimiter=',')
			print('wrote data to file')

		print(all_m_bins.shape)


		#---plot-mass-within-different-bins------------------------------------------------------------
		for j in np.arange(len(d_bins)-1):
			if i==0:
				d_in  = d_bins[j]
				d_out = d_bins[j+1]
				
				if np.isinf(d_out): d_out_str = r'$\infty$'
				else: d_out_str = str(d_out)
				this_label = str(d_in)+r' < $r$/kpc < '+d_out_str
				# axarr[j].set_yscale('log')
				# axarr[j].set_ylim(1e5,5e8)
			else:
				this_label = ''

			this_mstr = all_m_bins[:,j]
			this_mstr_cumulative = np.array([])
			m_cuml = 0
			for mass in this_mstr:
				m_cuml += mass
				this_mstr_cumulative = np.append(this_mstr_cumulative, m_cuml)

			this_mstr_normed = this_mstr_cumulative / np.amax(this_mstr_cumulative)

			axarr[j].plot(time,this_mstr_normed,lw=1.5,c=colors_list[i])
			axarr[j].annotate(this_label,xy=(0.95,0.1),xycoords='axes fraction',size=12,color='black',ha='right')

			if j==0:
				mlabel = r'; $M_\ast$ = ' + m.scinote(np.amax(this_mstr_cumulative)) + r' M$_\odot$'
				axarr[j].annotate(models_label[i]+mlabel,xy=(0.95,0.4-(i*0.1)),xycoords='axes fraction',size=12,color=colors_list[i],ha='right')
			else:
				mlabel = r'$M_\ast$ = ' + m.scinote(np.amax(this_mstr_cumulative)) + r' M$_\odot$'
				axarr[j].annotate(mlabel,xy=(0.95,0.4-(i*0.1)),xycoords='axes fraction',size=12,color=colors_list[i],ha='right')

		print('')
			

	#---finish-plotting----------------------------------------------------------------------------
	axarr[3].set_xlabel('time [Gyr]')
	axarr[3].set_xlim(0,2.86)

	# axarr[1].set_ylabel(r'$M_\ast$ [M$_\odot$]')
	axarr[1].set_ylabel(r'SFH')

	p.finalize(fig,'mstr_bins_'+whichsims,save=save)

def mstar_radial_birthpos(do_bins=True):
	fig,axarr = p.makefig('4_vert',figx=6.67,figy=10)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	for i,sim in enumerate(models):
		print(sim)
		axarr[0].annotate(models_label[i],xy=(0.07,0.85-(i*0.1)),xycoords='axes fraction',size=12,color=colors_list[i])

		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()

		mypath = d.smuggledir+sim
		onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
		a = onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)]
		a = np.sort(a)
		max_snap = int(a[-1].split('.')[0][-3:])

		#---calculate-mass-within-different-bins-------------------------------------------------------
		d_bins = np.array([0,0.2,0.5,1,np.inf])
		fname = d.datdir+'mstr_bins_'+sim+'.txt'

		snapfile = mypath+'/snapshot_400'
		# star_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h - CoM_all[400]
		staripos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h - CoM_all[400]
		starmass = snapHDF5.read_block(snapfile, 'MASS', parttype=4)*(1.e10)/h
		star_age = (time[400] - (snapHDF5.read_block(snapfile, 'AGE ', parttype=4)/h)) * 1.e3 # now in Myr

		d_star = np.linalg.norm(star_pos,axis=1)

		this_mstr = np.array([])
		for j in np.arange(len(d_bins)-1):
			d_in  = d_bins[j]
			d_out = d_bins[j+1]

			sel = (d_star >= d_in) & (d_star )


		print(all_m_bins.shape)


		#---plot-mass-within-different-bins------------------------------------------------------------
		for j in np.arange(len(d_bins)-1):
			if i==0:
				d_in  = d_bins[j]
				d_out = d_bins[j+1]
				
				if np.isinf(d_out): d_out_str = r'$\infty$'
				else: d_out_str = str(d_out)
				this_label = str(d_in)+r' < $r$/kpc < '+d_out_str
				axarr[j].set_yscale('log')
				axarr[j].set_ylabel(r'$M_\ast$ [M$_\odot$]')
				axarr[j].set_ylim(1e3,1e7)
			else:
				this_label = ''
			axarr[j].annotate(this_label,xy=(0.7,0.85),xycoords='axes fraction',size=12,color='black')

			this_mstr = all_m_bins[:,j]
			time_plt = time[:max_snap+1]
			if do_bins:
				dt = 0.05
				timebins = np.arange(0,np.amax(time_plt)+dt,dt)
				binwidth = (timebins[1] - timebins[0])/2

				mstr_mean = np.array([])
				for k in range(len(timebins)-1):
					leftbin = timebins[k]
					rightbin = timebins[k+1]

					sel = (time_plt > leftbin) & (time_plt < rightbin)
					mstr_mean = np.append(mstr_mean, np.mean(this_mstr[sel]))

				time_bins_plt = timebins[0:-1]+binwidth
				axarr[j].plot(time_bins_plt,mstr_mean,lw=1.5,color=colors_list[i])
				
			else:
				axarr[j].plot(time[:max_snap+1],this_mstr,lw=1.5,c=colors_list[i])
			
		print('----------------------------------------------\n\n')
			

	#---finish-plotting----------------------------------------------------------------------------
	axarr[3].set_xlabel('time [Gyr]')
	axarr[3].set_xlim(0,2.86)

	p.finalize(fig,'mstr_bins_'+sim,save=save)

def test_CoM_motion(sim):
	cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
	CoM_all = np.array(cm_file['CoM'])
	cm_file.close()

	print('CoM[400]: '+str(CoM_all[400][0])+', '+str(CoM_all[400][1])+', '+str(CoM_all[400][2]))
	print('CoM[000]: '+str(CoM_all[0][0])+', '+str(CoM_all[0][1])+', '+str(CoM_all[0][2]))
	print('difference: '+str(CoM_all[400][0] - CoM_all[0][0])+', '+str(CoM_all[400][1] - CoM_all[0][1])+', '+str(CoM_all[400][2] - CoM_all[0][2]))

def gas_rvr(sim,snapnum):
	i = np.where(models==sim)[0][0]

	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	text = ax.annotate(models_label[i],xy=(0.05,0.95),xycoords='axes fraction',size=12,color='white')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

	text = ax.annotate(r'$t$ = '+str(np.round(time[snapnum],2))+' Gyr',xy=(0.05,0.9),xycoords='axes fraction',size=12,color='white')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

	text = ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.05,0.85),xycoords='axes fraction',size=12,color='white')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

	text = ax.annotate('gas',xy=(0.8,0.9),xycoords='axes fraction',size=16,color='white')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

	cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
	CoM = np.array(cm_file['CoM'])[snapnum]
	cm_file.close()

	v_cm_file = h5py.File(d.datdir+'centerofmassvelocity_'+sim+'.hdf5','r')
	v_cm = np.array(v_cm_file['Vel_CoM'])[snapnum]
	v_cm_file.close()

	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
	gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h - CoM
	gas_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=0) - v_cm

	gas_r = np.linalg.norm(gas_pos,axis=1)
	sel = (gas_r >= 0.1) & (gas_r <= 10)
	gas_pos = gas_pos[sel]
	gas_vel = gas_vel[sel]
	gas_r = gas_r[sel]
	gas_vr = (gas_vel[:,0]*gas_pos[:,0] + gas_vel[:,1]*gas_pos[:,1] + gas_vel[:,2]*gas_pos[:,2]) / gas_r

	#---try-to-do-meshgrid-thing-------------------------------------------------------------------
	vr_bins = np.linspace(-100,100,100)
	# r_bins = np.log10(np.logspace(-1,1,100))
	r_bins = np.logspace(-1,1,100)

	do_manual_hist=True

	if do_manual_hist:

		for j in np.arange(len(vr_bins)-1):
			selj = (gas_vr >= vr_bins[j]) & (gas_vr < vr_bins[j+1])

			count_k_arr = np.array([])
			for k in np.arange(len(r_bins)-1):
				selk = (gas_r >= r_bins[k]) & (gas_r < r_bins[k+1])

				count = np.count_nonzero(selj & selk)
				count_k_arr = np.append(count_k_arr,count)

			if j==0:
				count_all_arr = count_k_arr
			else:
				count_all_arr = np.vstack((count_all_arr,count_k_arr))

		count_all_arr[count_all_arr==0] = 1e-10
		count_all_arr = np.log10(count_all_arr)
		r_bins = np.log10(r_bins)
		c = ax.pcolormesh(r_bins, vr_bins, count_all_arr, cmap='inferno', vmin=0, vmax=3)
		# ax.hist2d(np.log10(gas_r), gas_vr, bins=[r_bins,vr_bins], cmap=cm.inferno, normed=False)
	
	else:
		ax.plot(gas_r,gas_vr,'o',ms=2,mew=0,mfc='black',alpha=0.5)

	#---finish-plotting----------------------------------------------------------------------------

	# cbar = fig.colorbar(c, ax=ax,label=r'log$_{10}$(count)')

	# axcolor = 'white'
	# ax.spines['bottom'].set_color(axcolor)
	# ax.spines['top'].set_color(axcolor)
	# ax.spines['left'].set_color(axcolor)
	# ax.spines['right'].set_color(axcolor)
	# ax.tick_params(axis='both', which='both', color=axcolor)

	xmticks = np.log10(np.append(np.arange(0.1,1,0.1), np.arange(1,10,1)))

	# cbar = fig.colorbar(c)
	ax.set_xlim(-1,1)
	ax.set_xticks(np.log10(np.array([0.1,0.2,0.5,1,2,5,10])))
	ax.set_xticklabels(['0.1','0.2','0.5','1','2','5','10'])
	ax.set_xticks(xmticks,minor=True)
	ax.set_xlabel(r'$r$ [kpc]')

	ax.set_ylim(-100,100)
	ax.set_ylabel(r'$v_r$ [km s$^{-1}$]')

	# p.finalize(fig,'gas_rvr_'+sim+'_'+str(snapnum).zfill(3),save=save)
	fn = '/home/ejahn003/movie_frames/gas_rvr_'+sim+'/'+str(snapnum).zfill(3)+'.png'
	if save:
		print('saving: '+fn)
		plt.savefig(fn,format='png',dpi=150)
	else:
		print('showing: '+fn)
		plt.show()

def dark_rvr(sim,snapnum):
	i = np.where(models==sim)[0][0]

	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
	CoM = np.array(cm_file['CoM'])[snapnum]
	cm_file.close()

	v_cm_file = h5py.File(d.datdir+'centerofmassvelocity_'+sim+'.hdf5','r')
	v_cm = np.array(v_cm_file['Vel_CoM'])[snapnum]
	v_cm_file.close()
	
	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
	dark_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h - CoM
	dark_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=1) - v_cm

	dark_r = np.linalg.norm(dark_pos,axis=1)
	sel = (dark_r >= 0.1) & (dark_r <= 2)
	dark_pos = dark_pos[sel]
	dark_vel = dark_vel[sel]
	dark_r = dark_r[sel]
	dark_vr = (dark_vel[:,0]*dark_pos[:,0] + dark_vel[:,1]*dark_pos[:,1] + dark_vel[:,2]*dark_pos[:,2]) / dark_r

	# ax.plot(gas_r,gas_vr,'o',ms=2,mew=0,mfc='black',alpha=0.5)
	#---try-to-do-meshgrid-thing-------------------------------------------------------------------
	vr_bins = np.linspace(-100,100,100)
	# r_bins = np.log10(np.logspace(-1,0.3,100))
	r_bins = np.logspace(-1,0.3,100)

	do_2dhist = False
	do_manual_hist = False

	if do_2dhist:
		text = ax.annotate(models_label[i],xy=(0.05,0.95),xycoords='axes fraction',size=12,color='white')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		text = ax.annotate(r'$t$ = '+str(np.round(time[snapnum],2))+' Gyr',xy=(0.05,0.9),xycoords='axes fraction',size=12,color='white')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		text = ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.05,0.85),xycoords='axes fraction',size=12,color='white')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		text = ax.annotate('dark matter',xy=(0.7,0.9),xycoords='axes fraction',size=16,color='white')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

		r_bins = np.log10(r_bins)
		ax.hist2d(np.log10(dark_r), dark_vr, bins=[r_bins,vr_bins], cmap=cm.inferno, normed=True)

	elif do_manual_hist:	
		text = ax.annotate(models_label[i],xy=(0.05,0.95),xycoords='axes fraction',size=12,color='white')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		text = ax.annotate(r'$t$ = '+str(np.round(time[snapnum],2))+' Gyr',xy=(0.05,0.9),xycoords='axes fraction',size=12,color='white')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		text = ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.05,0.85),xycoords='axes fraction',size=12,color='white')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		text = ax.annotate('dark matter',xy=(0.7,0.9),xycoords='axes fraction',size=16,color='white')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

		for j in np.arange(len(vr_bins)-1):
			selj = (dark_vr >= vr_bins[j]) & (dark_vr < vr_bins[j+1])

			count_k_arr = np.array([])
			for k in np.arange(len(r_bins)-1):
				selk = (dark_r >= r_bins[k]) & (dark_r < r_bins[k+1])

				count = np.count_nonzero(selj & selk)
				count_k_arr = np.append(count_k_arr,count)

			if j==0:
				count_all_arr = count_k_arr
			else:
				count_all_arr = np.vstack((count_all_arr,count_k_arr))

		count_all_arr[count_all_arr==0] = 1e-10
		count_all_arr = np.log10(count_all_arr)
		r_bins = np.log10(r_bins)
		c = ax.pcolormesh(r_bins, vr_bins, count_all_arr, cmap='inferno', vmin=0, vmax=3)
		# cbar = fig.colorbar(c, ax=ax,label=r'log$_{10}$(count)')

	else:
		text = ax.annotate(models_label[i],xy=(0.95,0.15),xycoords='axes fraction',ha='right',size=12,color='black')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])
		text = ax.annotate(r'$t$ = '+str(np.round(time[snapnum],2))+' Gyr',xy=(0.95,0.10),xycoords='axes fraction',ha='right',size=12,color='black')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])
		text = ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.95,0.05),xycoords='axes fraction',ha='right',size=12,color='black')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])
		text = ax.annotate('dark matter',xy=(0.95,0.9),xycoords='axes fraction',ha='right',size=16,color='black')
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='black', edgecolor='white'), path_effects.Normal()])

		ax.plot(np.log10(dark_r),dark_vr,'o',ms=1,mew=0,mfc='black',alpha=1)

	#---finish-plotting----------------------------------------------------------------------------
	
	# axcolor = 'white'
	# ax.spines['bottom'].set_color(axcolor)
	# ax.spines['top'].set_color(axcolor)
	# ax.spines['left'].set_color(axcolor)
	# ax.spines['right'].set_color(axcolor)
	# ax.tick_params(axis='both', which='both', color=axcolor)

	# xmticks = np.log10(np.append(np.arange(0.1,1,0.1), np.arange(1,10,1)))
	xmticks = np.log10(np.arange(0.1,1,0.1))
	ax.set_xlim(-1,0.3)
	ax.set_xticks(np.log10(np.array([0.1,0.2,0.5,1,2])))
	ax.set_xticklabels(['0.1','0.2','0.5','1','2'])
	ax.set_xticks(xmticks,minor=True)
	ax.set_xlabel(r'$r$ [kpc]')

	ax.set_ylim(-200,200)
	ax.set_ylabel(r'$v_r$ [km s$^{-1}$]')

	# p.finalize(fig,'dark_rvr_'+sim+'_'+str(snapnum).zfill(3),save=save)
	# fn = '/home/ejahn003/movie_frames/dark_rvr_'+sim+'/'+str(snapnum).zfill(3)+'.png'
	fn = d.plotdir+month+'/dark_rvr_'+sim+'_'+str(snapnum).zfill(3)+'.png'

	if save:
		print('saving: '+fn)
		plt.savefig(fn,format='png',dpi=150)
	else:
		print('showing: '+fn)
		plt.show()

def str_rvr(sim,snapnum):
	i = np.where(models==sim)[0][0]

	fig,ax = p.makefig(1,figx=6.67,figy=5)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()

	text = ax.annotate(models_label[i],xy=(0.05,0.95),xycoords='axes fraction',size=12,color='white')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

	text = ax.annotate(r'$t$ = '+str(np.round(time[snapnum],2))+' Gyr',xy=(0.05,0.9),xycoords='axes fraction',size=12,color='white')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

	text = ax.annotate('snapshot '+str(snapnum).zfill(3),xy=(0.05,0.85),xycoords='axes fraction',size=12,color='white')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

	text = ax.annotate('stars',xy=(0.8,0.9),xycoords='axes fraction',size=16,color='white')
	text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])

	cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
	CoM = np.array(cm_file['CoM'])[snapnum]
	cm_file.close()

	v_cm_file = h5py.File(d.datdir+'centerofmassvelocity_'+sim+'.hdf5','r')
	v_cm = np.array(v_cm_file['Vel_CoM'])[snapnum]
	v_cm_file.close()

	snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
	str_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=4)/h - CoM
	str_vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=4) - v_cm

	try:
		str_r = np.linalg.norm(str_pos,axis=1)
		sel = (str_r >= 0.1) & (str_r <= 10)
		str_pos = str_pos[sel]
		str_vel = str_vel[sel]
		str_r = str_r[sel]
		str_vr = (str_vel[:,0]*str_pos[:,0] + str_vel[:,1]*str_pos[:,1] + str_vel[:,2]*str_pos[:,2]) / str_r
		has_type4 = True
	except:
		has_type4 = False

	#---try-to-do-meshgrid-thing-------------------------------------------------------------------
	vr_bins = np.linspace(-150,150,100)
	# r_bins = np.log10(np.logspace(-1,1,100))
	r_bins = np.logspace(-1,1,100)

	do_manual_hist=True

	if do_manual_hist:
		if has_type4:
			for j in np.arange(len(vr_bins)-1):
				selj = (str_vr >= vr_bins[j]) & (str_vr < vr_bins[j+1])

				count_k_arr = np.array([])
				for k in np.arange(len(r_bins)-1):
					selk = (str_r >= r_bins[k]) & (str_r < r_bins[k+1])

					count = np.count_nonzero(selj & selk)
					count_k_arr = np.append(count_k_arr,count)

				if j==0:
					count_all_arr = count_k_arr
				else:
					count_all_arr = np.vstack((count_all_arr,count_k_arr))

			count_all_arr[count_all_arr==0] = 1e-10
			count_all_arr = np.log10(count_all_arr)
		else:
			count_all_arr = np.zeros((100,100)) + 1e-10

		r_bins = np.log10(r_bins)
		c = ax.pcolormesh(r_bins, vr_bins, count_all_arr, cmap='inferno', vmin=0, vmax=3)
		# ax.hist2d(np.log10(gas_r), gas_vr, bins=[r_bins,vr_bins], cmap=cm.inferno, normed=False)
	
	else:
		ax.plot(str_r,str_vr,'o',ms=2,mew=0,mfc='black',alpha=0.5)

	#---finish-plotting----------------------------------------------------------------------------

	# cbar = fig.colorbar(c, ax=ax,label=r'log$_{10}$(count)')

	# axcolor = 'white'
	# ax.spines['bottom'].set_color(axcolor)
	# ax.spines['top'].set_color(axcolor)
	# ax.spines['left'].set_color(axcolor)
	# ax.spines['right'].set_color(axcolor)
	# ax.tick_params(axis='both', which='both', color=axcolor)

	xmticks = np.log10(np.append(np.arange(0.1,1,0.1), np.arange(1,10,1)))

	# cbar = fig.colorbar(c)
	ax.set_xlim(-1,1)
	ax.set_xticks(np.log10(np.array([0.1,0.2,0.5,1,2,5,10])))
	ax.set_xticklabels(['0.1','0.2','0.5','1','2','5','10'])
	ax.set_xticks(xmticks,minor=True)
	ax.set_xlabel(r'$r$ [kpc]')

	ax.set_ylim(-150,150)
	ax.set_ylabel(r'$v_r$ [km s$^{-1}$]')

	# p.finalize(fig,'gas_rvr_'+sim+'_'+str(snapnum).zfill(3),save=save)
	fn = '/home/ejahn003/movie_frames/str_rvr_'+sim+'/'+str(snapnum).zfill(3)+'.png'
	if save:
		print('saving: '+fn)
		plt.savefig(fn,format='png',dpi=150)
	else:
		print('showing: '+fn)
		plt.show()

def sfrcolor_single(sim):
	print('plotting time_radius_sfr_colormesh '+sim)
	i = np.where(models==sim)[0][0]
	dist_bins = np.logspace(-2,np.log10(5),100)

	mypath = d.smuggledir+sim
	onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
	a = np.sort(onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)])
	max_snap = int(a[-1].split('.')[0][-3:])

	cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
	CoM_all = np.array(cm_file['CoM'])
	cm_file.close()

	tf = h5py.File(d.datdir+'timefile.hdf5','r')
	time = np.array(tf['time'])
	tf.close()
	if max_snap == 400:
		dt = time[-1] - time[-2]
		time = np.append(time,np.amax(time)+dt)
	else:
		time = time[:max_snap+1]

	for snapnum in np.arange(0,max_snap+1):
		printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
		sys.stdout.write(printthing); sys.stdout.flush()
		if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
		else: print('')

		snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
		gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h - CoM_all[snapnum]
		gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)

		sel_in_bound = (np.abs(gas_pos[:,0]) < 5) & (np.abs(gas_pos[:,1]) < 5) & (np.abs(gas_pos[:,2]) < 5) & (gas_sfr > 0) & np.logical_not(np.isnan(gas_sfr))
		gas_pos = gas_pos[sel_in_bound]
		gas_sfr = gas_sfr[sel_in_bound]

		d_gas = np.linalg.norm(gas_pos,axis=1)

		sfr_profile = np.array([])
		for j in np.arange(len(dist_bins)-1):
			seld = (d_gas > dist_bins[j]) & (d_gas < dist_bins[j+1])
			sfr_profile = np.append(sfr_profile, np.sum(gas_sfr[seld]))

		sel = (sfr_profile <= 0) & np.logical_not(np.isnan(sfr_profile)) & np.logical_not(np.isinf(sfr_profile))
		sfr_profile[sel] = 1e-10

		if snapnum == 0:
			sfr_profile_all = sfr_profile
		else:
			sfr_profile_all = np.vstack((sfr_profile_all,sfr_profile))

	# x,y = np.meshgrid(np.arange(402),dist_bins)
	x,y = np.meshgrid(time,dist_bins)
	z = np.log10(sfr_profile_all.T)
	z_min = np.amin(z[z>-10])
	z_max = np.amax(z)
	print('min log sfr: '+str(z_min))
	print('max log sfr: '+str(z_max))

	print(x.shape)
	print(y.shape)
	print(z.shape)

	fig,ax = p.makefig(1,figx=10,figy=3)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	c = ax.pcolormesh(x, y, z, cmap='inferno', vmin=-6, vmax=0)

	# ax.set_xlabel('snapshots')
	# ax.set_xlim(0,400)

	ax.set_xlim(0,2.86)
	ax.set_xlabel('time [Gyr]')
	
	ax.set_ylabel('distance [kpc]')
	ax.set_yscale('log')
	ax.set_ylim(0.01,5)
	ax.set_yticks([0.01,0.1,1,5])
	ax.set_yticklabels(['0.01','0.1','1','5'])

	cbar = fig.colorbar(c, ax=ax,label=r'log$_{10}$(SFR)',ticks=[-6,-5,-4,-3,-2,-1,0])

	if max_snap < 360:
		ax.annotate(models_label[i],xy=(0.7,0.1),xycoords='axes fraction',color='black',size=12)
	else:
		ax.annotate(models_label[i],xy=(0.7,0.1),xycoords='axes fraction',color='white',size=12)
	# cbar.ax.set_yticks[]
	p.finalize(fig,'SFR_time_radius_'+sim,save=save)

def sfrcolor_panels(write=True,do_r50=True,do_core=True,do_Mstar=False):
	print('plotting time_radius_sfrcolor_panels')

	if len(models)==3:
		fig,axarr = p.makefig('3_vert',figx=10,figy=10)
	elif len(models)==7:
		fig,axarr = p.makefig('7_vert',figx=10,figy=20.5)
	else:
		raise ValueError('len of models is wrong. needs to be 3 or 7')

	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	# fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)


	dist_bins = np.logspace(-2,np.log10(5),100)

	axcolor = 'white'
	axlist = axarr

	an_x_pos = 0.35

	if do_Mstar:
		text = axarr[0].annotate(r'$M_\ast$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='orange',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_core:
		text = axarr[0].annotate(r'$r_\mathregular{core}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='turquoise',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_r50:
		text = axarr[0].annotate(r'$r_\mathregular{50, SF gas}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='white',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		

	for i,sim in enumerate(models):
		print(sim)
		text = axarr[i].annotate(models_label[i],xy=(0.05,0.8),xycoords='axes fraction',color='white',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='white', edgecolor='black'), path_effects.Normal()])
		fname = d.datdir+'/sfr_profile_all_'+sim+'.txt'

		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		mypath = d.smuggledir+sim
		onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
		a = np.sort(onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)])
		max_snap = int(a[-1].split('.')[0][-3:])
		print('max_snap = '+str(max_snap))

		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()

		#---plot-SFR-profile-colors----------------------------------------------------------------
		try:
			sfr_profile_all = np.loadtxt(fname,dtype=float,delimiter=',')
			print('successfully read sfr profiles data from file')
			preexist = True
		except:
			print('could not read data from file. calculating sfr profiles manually ')
			preexist = False

			for snapnum in np.arange(0,max_snap+1):
				printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
				sys.stdout.write(printthing); sys.stdout.flush()
				if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
				else: print('')

				snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
				gas_pos = snapHDF5.read_block(snapfile, 'POS ', parttype=0)/h - CoM_all[snapnum]
				gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)

				sel_in_bound = (np.abs(gas_pos[:,0]) < 5) & (np.abs(gas_pos[:,1]) < 5) & (np.abs(gas_pos[:,2]) < 5) & (gas_sfr > 0) & np.logical_not(np.isnan(gas_sfr))
				gas_pos = gas_pos[sel_in_bound]
				gas_sfr = gas_sfr[sel_in_bound]

				d_gas = np.linalg.norm(gas_pos,axis=1)

				sfr_profile = np.array([])
				for j in np.arange(len(dist_bins)-1):
					seld = (d_gas > dist_bins[j]) & (d_gas < dist_bins[j+1])
					sfr_profile = np.append(sfr_profile, np.sum(gas_sfr[seld]))

				sel = (sfr_profile <= 0) & np.logical_not(np.isnan(sfr_profile)) & np.logical_not(np.isinf(sfr_profile))
				sfr_profile[sel] = 1e-10

				if snapnum == 0:
					sfr_profile_all = sfr_profile
				else:
					sfr_profile_all = np.vstack((sfr_profile_all,sfr_profile))
		if write:
			print('saving sfr_profile_all to '+fname)
			np.savetxt(fname,sfr_profile_all,delimiter=',')

		print('plotting SFR color')
		x,y = np.meshgrid(time,dist_bins)

		if max_snap < 400:
			new_zs = np.zeros((400 - max_snap, len(dist_bins) - 1)) - 5
			sfr_profile_all = np.vstack((sfr_profile_all,new_zs))

		z = np.log10(sfr_profile_all.T)
		c = axarr[i].pcolormesh(x, y, z, cmap='inferno', vmin=-4, vmax=-0.5)

		#---plot-horizontal-lines------------------------------------------------------------------
		axarr[i].axhline(y=0.2,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=0.5,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=1,ls='--',color='white',lw=0.7,alpha=0.3)

		#---plot-core-radius-----------------------------------------------------------------------
		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		use_nfw_core = True

		if do_core:
			print('plotting core radius')
			if use_nfw_core:
				try: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3_hires.hdf5','r')
				except: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3.hdf5','r')
				
				core_radius = np.array(nf['core_radius'])
				nf.close()
				print('using NFW core')
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='black',lw=4)
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='turquoise',lw=1)
			else:
				try:
					nf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5','r')
				except:
			 		nf = h5py.File(d.datdir+'PITS_params_'+sim+'.hdf5','r')
				core_radius = np.array(nf['rc'])
				nf.close()
				print('using PITS core')

				dt = 0.05
				timebins = np.arange(0,np.amax(time)+dt,dt)
				binwidth = (timebins[1] - timebins[0])/2

				rcore_mean = np.array([])
				for j in range(len(timebins)-1):
					leftbin = timebins[j]
					rightbin = timebins[j+1]

					selbin = (time > leftbin) & (time < rightbin)
					rcore_mean = np.append(rcore_mean, np.mean(core_radius[selbin]))

				time_plt = timebins[0:-1]+binwidth
				axarr[i].plot(time_plt,rcore_mean,color='turquoise',lw=0.7)

				# axarr[i].plot(time,core_radius,'-',c='turquoise',lw=0.7)

		#---plot-r50SFgas--------------------------------------------------------------------------
		if do_r50:
			print('plotting r50')
			f = h5py.File(d.datdir+'r50SFgas_'+sim+'.hdf5','r')
			all_r50 = np.array(f['all_r50'])
			f.close()

			fn = d.datdir+'num_sfr_time_'+sim+'txt'
			try:
				print('read num_sfr from file')
				num_sfr = np.loadtxt(fn,dtype=float,delimiter=',')
			except:
				print('calculating num_sfr')
				num_sfr = np.array([])
				for snapnum in np.arange(0,max_snap+1):
					printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
					sys.stdout.write(printthing); sys.stdout.flush()
					if not(snapnum==max_snap+1): sys.stdout.write("\b" * (len(printthing)))
					else: print('')
					snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
					gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
					num_sfr = np.append(num_sfr, np.count_nonzero(gas_sfr > 0))
				np.savetxt(fn,num_sfr,delimiter=',')
				print('wrote num_sfr to '+fn)

			sel = num_sfr <= 10
			all_r50[sel] = np.nan

			time = time[:max_snap+1]

			dt = 0.025
			timebins = np.arange(0,np.amax(time)+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2

			r50_mean = np.array([])
			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				selbin = (time > leftbin) & (time < rightbin) & np.logical_not(np.isnan(all_r50))
				r50_mean = np.append(r50_mean, np.mean(all_r50[selbin]))

			time_plt = timebins[0:-1]+binwidth
			axarr[i].plot(time_plt,r50_mean,color='black',lw=2)
			axarr[i].plot(time_plt,r50_mean,color='white',lw=0.7)

		#---plot-cumulative-Mstar------------------------------------------------------------------
		if do_Mstar:
			print('plotting M_star')
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
			type4_profile_all = np.array(f['type4'])
			f.close()

			M_star_all = np.array([])
			for profile in type4_profile_all:
				M_star_all = np.append(M_star_all,profile[-1])

			M_star_all = M_star_all/(1.e7)

			ax2 = axarr[i].twinx()
			ax2.plot(time,M_star_all,c='orange',lw=1)
			ax2.set_ylim(0,10)

			if i==0: ax2.set_yticks([0,2,4,6,8,10])
			else: ax2.set_yticks([0,2,4,6,8])
			ax2.set_yticks(np.arange(10),minor=True)

			# ax2.set_yscale('log')
			ax2.set_xlim(0,2.86)
			ax2.tick_params(axis='both', which='both', color='white')
			if i==1: ax2.set_ylabel(r'$M_\ast$ [$10^7$ M$_\odot$]')
			axlist = np.append(axlist,ax2)

		print('-----------------------------------------------------------------\n')

	#---finish-setting-up-plot---------------------------------------------------------------------
	for ax in axlist:
		ax.spines['bottom'].set_color(axcolor)
		ax.spines['top'].set_color(axcolor)
		ax.spines['left'].set_color(axcolor)
		ax.spines['right'].set_color(axcolor)
		ax.tick_params(axis='both', which='both', color=axcolor)

	do_horiz_cbar = True

	if do_horiz_cbar:
		if len(models)==3: yanch = 10.0
		elif len(models)==7: yanch = 8.0
		cbar = fig.colorbar(c, anchor=(0.0,yanch), orientation='horizontal', ax=axlist.ravel().tolist(),
			ticks=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5],label=r'log$_{10}$(SFR [M$_\odot$ yr$^{-1}$])')
		cbar.ax.xaxis.set_ticks_position('top')
		cbar.ax.xaxis.set_label_position('top')

	else:
		cbar = fig.colorbar(c, ax=axlist.ravel().tolist(),
			ticks=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5],label=r'log$_{10}$(SFR [M$_\odot$ yr$^{-1}$])')

	axarr[6].set_xlim(0,2.86)
	axarr[6].set_xlabel('time [Gyr]')
	axarr[6].set_xticks([0,0.5,1,1.5,2,2.5])
	axarr[6].set_xticks(np.arange(0,2.86,0.1),minor=True)	
	axarr[6].set_xticklabels(['0','0.5','1','1.5','2','2.5'])

	yminticks = np.append(np.arange(0.1,1,0.1),np.arange(1,3,1))
	# axarr[0].set_ylabel('distance [kpc]')
	axarr[0].set_yscale('log')
	axarr[0].set_ylim(0.1,5)
	axarr[0].set_yticks([0.1,0.2,0.5,1,2,5])
	axarr[0].set_yticklabels(['0.1','0.2','0.5','1','2','5'])
	axarr[0].set_yticks(yminticks,minor=True)

	axarr[1].set_ylabel('distance [kpc]')
	for j in np.arange(1,len(models)):
		axarr[j].set_yscale('log')
		axarr[j].set_ylim(0.1,5)
		axarr[j].set_yticks([0.1,0.2,0.5,1,2])
		axarr[j].set_yticklabels(['0.1','0.2','0.5','1','2'])
		axarr[j].set_yticks(yminticks,minor=True)

	# axarr[2].set_yscale('log')
	# axarr[2].set_ylim(0.1,5)
	# axarr[2].set_yticks([0.1,0.2,0.5,1,2])
	# axarr[2].set_yticklabels(['0.1','0.2','0.5','1','2'])
	# axarr[2].set_yticks(yminticks,minor=True)


	# cbar.ax.set_yticks[]
	p.finalize(fig,'SFRcolor_'+whichsims,save=save,tight=True)
	# if save:
	# 	plt.savefig(d.plotdir+month+'/SFRcolor_time_radius_panels_'+whichsims+'.png',format='png',dpi=200)
	# else:
	# 	plt.show()

def gasfrac_color_panels(do_r50=True,do_core=True,do_Mstar=False):
	print('plotting time_radius_sfrcolor_panels')
	
	if len(models)==3:
		fig,axarr = p.makefig('3_vert',figx=10,figy=10)
	elif len(models)==7:
		fig,axarr = p.makefig('7_vert',figx=10,figy=20.5)
	else:
		raise ValueError('len of models is wrong. needs to be 3 or 7')
		
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)

	axcolor = 'white'
	axlist = axarr

	an_x_pos = 0.35

	if do_Mstar:
		text = axarr[0].annotate(r'$M_\ast$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='orange',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_core:
		text = axarr[0].annotate(r'$r_\mathregular{core}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='turquoise',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_r50:
		text = axarr[0].annotate(r'$r_\mathregular{50, SF gas}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='white',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		

	for i,sim in enumerate(models):
		print(sim)
		text = axarr[i].annotate(models_label[i],xy=(0.05,0.8),xycoords='axes fraction',color='white',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='white', edgecolor='black'), path_effects.Normal()])
		fname = d.datdir+'/gasfrac_profile_all_'+sim+'.txt'

		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		try:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		except:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')

		drange = np.array(f['drange'])
		gas_profile_all = np.array(f['gas'])
		dark_profile_all = np.array(f['dark'])
		f.close()
		
		max_snap = dark_profile_all.shape[0] - 1
		print('max_snap = '+str(max_snap))

		sel = (drange > 0.1) & (drange < 5)
		gas_profile_all = gas_profile_all[:,sel]
		dark_profile_all = dark_profile_all[:,sel]
		dist_bins = drange[sel]
		
		frac_profile_all = gas_profile_all / dark_profile_all

		print('plotting gasfrac color')
		print('max frac: '+str(np.amax(frac_profile_all)))
		print('min frac: '+str(np.amin(frac_profile_all[frac_profile_all > 0])))

		if max_snap < 400:
			new_zs = np.zeros((400 - max_snap, len(dist_bins))) - 5
			frac_profile_all = np.vstack((frac_profile_all,new_zs))

		x,y = np.meshgrid(time,dist_bins)
		z = np.log10(frac_profile_all.T)
		c = axarr[i].pcolormesh(x, y, z, cmap='inferno', vmin=-2, vmax=0)

		#---plot-horizontal-lines------------------------------------------------------------------
		axarr[i].axhline(y=0.2,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=0.5,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=1,ls='--',color='white',lw=0.7,alpha=0.3)

		#---plot-core-radius-----------------------------------------------------------------------
		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		use_nfw_core = True

		if do_core:
			print('plotting core radius')
			if use_nfw_core:
				try: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3_hires.hdf5','r')
				except: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3.hdf5','r')
				
				core_radius = np.array(nf['core_radius'])
				nf.close()
				print('using NFW core')
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='black',lw=4)
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='turquoise',lw=1)
			else:
				try:
					nf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5','r')
				except:
			 		nf = h5py.File(d.datdir+'PITS_params_'+sim+'.hdf5','r')
				core_radius = np.array(nf['rc'])
				nf.close()
				print('using PITS core')

				dt = 0.05
				timebins = np.arange(0,np.amax(time)+dt,dt)
				binwidth = (timebins[1] - timebins[0])/2

				rcore_mean = np.array([])
				for j in range(len(timebins)-1):
					leftbin = timebins[j]
					rightbin = timebins[j+1]

					selbin = (time > leftbin) & (time < rightbin)
					rcore_mean = np.append(rcore_mean, np.mean(core_radius[selbin]))

				time_plt = timebins[0:-1]+binwidth
				axarr[i].plot(time_plt,rcore_mean,color='turquoise',lw=0.7)

				# axarr[i].plot(time,core_radius,'-',c='turquoise',lw=0.7)

		#---plot-r50SFgas--------------------------------------------------------------------------
		if do_r50:
			print('plotting r50')
			f = h5py.File(d.datdir+'r50SFgas_'+sim+'.hdf5','r')
			all_r50 = np.array(f['all_r50'])
			f.close()

			fn = d.datdir+'num_sfr_time_'+sim+'txt'
			try:
				print('read num_sfr from file')
				num_sfr = np.loadtxt(fn,dtype=float,delimiter=',')
			except:
				print('calculating num_sfr')
				num_sfr = np.array([])
				for snapnum in np.arange(0,max_snap+1):
					printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
					sys.stdout.write(printthing); sys.stdout.flush()
					if not(snapnum==max_snap+1): sys.stdout.write("\b" * (len(printthing)))
					else: print('')
					snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
					gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
					num_sfr = np.append(num_sfr, np.count_nonzero(gas_sfr > 0))
				np.savetxt(fn,num_sfr,delimiter=',')
				print('wrote num_sfr to '+fn)

			sel = num_sfr <= 10
			all_r50[sel] = np.nan

			time = time[:max_snap+1]

			dt = 0.025
			timebins = np.arange(0,np.amax(time)+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2

			r50_mean = np.array([])
			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				selbin = (time > leftbin) & (time < rightbin) & np.logical_not(np.isnan(all_r50))
				r50_mean = np.append(r50_mean, np.mean(all_r50[selbin]))

			time_plt = timebins[0:-1]+binwidth
			axarr[i].plot(time_plt,r50_mean,color='black',lw=2)
			axarr[i].plot(time_plt,r50_mean,color='white',lw=0.7)

		#---plot-cumulative-Mstar------------------------------------------------------------------
		if do_Mstar:
			print('plotting M_star')
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
			type4_profile_all = np.array(f['type4'])
			f.close()

			M_star_all = np.array([])
			for profile in type4_profile_all:
				M_star_all = np.append(M_star_all,profile[-1])

			M_star_all = M_star_all/(1.e7)

			ax2 = axarr[i].twinx()
			ax2.plot(time,M_star_all,c='orange',lw=1)
			ax2.set_ylim(0,10)

			if i==0: ax2.set_yticks([0,2,4,6,8,10])
			else: ax2.set_yticks([0,2,4,6,8])
			ax2.set_yticks(np.arange(10),minor=True)

			# ax2.set_yscale('log')
			ax2.set_xlim(0,2.86)
			ax2.tick_params(axis='both', which='both', color='white')
			if i==1: ax2.set_ylabel(r'$M_\ast$ [$10^7$ M$_\odot$]')
			axlist = np.append(axlist,ax2)

		print('-----------------------------------------------------------------\n')

	#---finish-setting-up-plot---------------------------------------------------------------------
	for ax in axlist:
		ax.spines['bottom'].set_color(axcolor)
		ax.spines['top'].set_color(axcolor)
		ax.spines['left'].set_color(axcolor)
		ax.spines['right'].set_color(axcolor)
		ax.tick_params(axis='both', which='both', color=axcolor)

	do_horiz_cbar = True

	if do_horiz_cbar:
		if len(models)==3: yanch = 10.0
		elif len(models)==7: yanch = 8.0
		cbar = fig.colorbar(c, anchor=(0.0,yanch), orientation='horizontal', ax=axlist.ravel().tolist(),
			ticks=[-2,-1.5,-1,-0.5,0],label=r'log$_{10}$($M_\mathregular{gas}$ / $M_\mathregular{DM}$)')
		cbar.ax.xaxis.set_ticks_position('top')
		cbar.ax.xaxis.set_label_position('top')

	else:
		cbar = fig.colorbar(c, ax=axlist.ravel().tolist(),
			ticks=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5],label=r'log$_{10}$(SFR [M$_\odot$ yr$^{-1}$])')

	axarr[6].set_xlim(0,2.86)
	axarr[6].set_xlabel('time [Gyr]')
	axarr[6].set_xticks([0,0.5,1,1.5,2,2.5])
	axarr[6].set_xticks(np.arange(0,2.86,0.1),minor=True)	
	axarr[6].set_xticklabels(['0','0.5','1','1.5','2','2.5'])

	yminticks = np.append(np.arange(0.1,1,0.1),np.arange(1,3,1))
	# axarr[0].set_ylabel('distance [kpc]')
	axarr[0].set_yscale('log')
	axarr[0].set_ylim(0.1,5)
	axarr[0].set_yticks([0.1,0.2,0.5,1,2,5])
	axarr[0].set_yticklabels(['0.1','0.2','0.5','1','2','5'])
	axarr[0].set_yticks(yminticks,minor=True)

	axarr[1].set_ylabel('distance [kpc]')
	for j in np.arange(1,len(models)):
		axarr[j].set_yscale('log')
		axarr[j].set_ylim(0.1,5)
		axarr[j].set_yticks([0.1,0.2,0.5,1,2])
		axarr[j].set_yticklabels(['0.1','0.2','0.5','1','2'])
		axarr[j].set_yticks(yminticks,minor=True)


	# cbar.ax.set_yticks[]
	p.finalize(fig,'gasfrac_color_'+whichsims,save=save,tight=True)
	# if save:
	# 	plt.savefig(d.plotdir+month+'/SFRcolor_time_radius_panels_'+whichsims+'.png',format='png',dpi=200)
	# else:
	# 	plt.show()

def starfrac_color_panels(do_r50=True,do_core=True,do_Mstar=False):
	print('plotting time_radius_sfrcolor_panels')
	
	# if not(whichsims=='ff_eSF_var_1e6' or whichsims=='eSF_eSFramp_eSF2_1e6'): 
	# 	raise ValueError('please use whichsims = ff_eSF_var_1e6 or eSF_eSFramp_eSF2_1e6')

	if len(models)==3:
		fig,axarr = p.makefig('3_vert',figx=10,figy=10)
	elif len(models)==7:
		fig,axarr = p.makefig('7_vert',figx=10,figy=20.5)
	else:
		raise ValueError('len of models is wrong. needs to be 3 or 7')
	
	dist_bins = np.logspace(-2,np.log10(5),100)
	
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)

	axcolor = 'white'
	axlist = axarr

	an_x_pos = 0.35

	if do_Mstar:
		text = axarr[0].annotate(r'$M_\ast$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='orange',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_core:
		text = axarr[0].annotate(r'$r_\mathregular{core}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='turquoise',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_r50:
		text = axarr[0].annotate(r'$r_\mathregular{50, SF gas}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='white',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		

	for i,sim in enumerate(models):
		print(sim)
		text = axarr[i].annotate(models_label[i],xy=(0.05,0.8),xycoords='axes fraction',color='white',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='white', edgecolor='black'), path_effects.Normal()])

		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		try:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		except:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')

		drange = np.array(f['drange'])
		star_profile_all = np.array(f['type4']) + np.array(f['type3']) + np.array(f['type2'])
		dark_profile_all = np.array(f['dark'])
		f.close()
		
		max_snap = dark_profile_all.shape[0] - 1
		print('max_snap = '+str(max_snap))

		sel = (drange > 0.1) & (drange < 5)
		star_profile_all = star_profile_all[:,sel]
		dark_profile_all = dark_profile_all[:,sel]
		dist_bins = drange[sel]
		
		frac_profile_all = star_profile_all / dark_profile_all

		print('plotting gasfrac color')
		print('max frac: '+str(np.amax(frac_profile_all)))
		print('min frac: '+str(np.amin(frac_profile_all[frac_profile_all > 0])))

		if max_snap < 400:
			new_zs = np.zeros((400 - max_snap, len(dist_bins))) - 5
			frac_profile_all = np.vstack((frac_profile_all,new_zs))

		x,y = np.meshgrid(time,dist_bins)
		# z = np.log10(frac_profile_all.T)
		z = frac_profile_all.T
		c = axarr[i].pcolormesh(x, y, z, cmap='inferno', vmin=0, vmax=1)

		#---plot-horizontal-lines------------------------------------------------------------------
		axarr[i].axhline(y=0.2,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=0.5,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=1,ls='--',color='white',lw=0.7,alpha=0.3)

		#---plot-core-radius-----------------------------------------------------------------------
		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		use_nfw_core = True

		if do_core:
			print('plotting core radius')
			if use_nfw_core:
				try: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3_hires.hdf5','r')
				except: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3.hdf5','r')
				
				core_radius = np.array(nf['core_radius'])
				nf.close()
				print('using NFW core')
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='black',lw=4)
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='turquoise',lw=1)
			else:
				try:
					nf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5','r')
				except:
			 		nf = h5py.File(d.datdir+'PITS_params_'+sim+'.hdf5','r')
				core_radius = np.array(nf['rc'])
				nf.close()
				print('using PITS core')

				dt = 0.05
				timebins = np.arange(0,np.amax(time)+dt,dt)
				binwidth = (timebins[1] - timebins[0])/2

				rcore_mean = np.array([])
				for j in range(len(timebins)-1):
					leftbin = timebins[j]
					rightbin = timebins[j+1]

					selbin = (time > leftbin) & (time < rightbin)
					rcore_mean = np.append(rcore_mean, np.mean(core_radius[selbin]))

				time_plt = timebins[0:-1]+binwidth
				axarr[i].plot(time_plt,rcore_mean,color='turquoise',lw=0.7)

				# axarr[i].plot(time,core_radius,'-',c='turquoise',lw=0.7)

		#---plot-r50SFgas--------------------------------------------------------------------------
		if do_r50:
			print('plotting r50')
			f = h5py.File(d.datdir+'r50SFgas_'+sim+'.hdf5','r')
			all_r50 = np.array(f['all_r50'])
			f.close()

			fn = d.datdir+'num_sfr_time_'+sim+'txt'
			try:
				print('read num_sfr from file')
				num_sfr = np.loadtxt(fn,dtype=float,delimiter=',')
			except:
				print('calculating num_sfr')
				num_sfr = np.array([])
				for snapnum in np.arange(0,max_snap+1):
					printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
					sys.stdout.write(printthing); sys.stdout.flush()
					if not(snapnum==max_snap+1): sys.stdout.write("\b" * (len(printthing)))
					else: print('')
					snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
					gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
					num_sfr = np.append(num_sfr, np.count_nonzero(gas_sfr > 0))
				np.savetxt(fn,num_sfr,delimiter=',')
				print('wrote num_sfr to '+fn)

			sel = num_sfr <= 10
			all_r50[sel] = np.nan

			time = time[:max_snap+1]

			dt = 0.025
			timebins = np.arange(0,np.amax(time)+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2

			r50_mean = np.array([])
			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				selbin = (time > leftbin) & (time < rightbin) & np.logical_not(np.isnan(all_r50))
				r50_mean = np.append(r50_mean, np.mean(all_r50[selbin]))

			time_plt = timebins[0:-1]+binwidth
			axarr[i].plot(time_plt,r50_mean,color='black',lw=2)
			axarr[i].plot(time_plt,r50_mean,color='white',lw=0.7)

		#---plot-cumulative-Mstar------------------------------------------------------------------
		if do_Mstar:
			print('plotting M_star')
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
			type4_profile_all = np.array(f['type4'])
			f.close()

			M_star_all = np.array([])
			for profile in type4_profile_all:
				M_star_all = np.append(M_star_all,profile[-1])

			M_star_all = M_star_all/(1.e7)

			ax2 = axarr[i].twinx()
			ax2.plot(time,M_star_all,c='orange',lw=1)
			ax2.set_ylim(0,10)

			if i==0: ax2.set_yticks([0,2,4,6,8,10])
			else: ax2.set_yticks([0,2,4,6,8])
			ax2.set_yticks(np.arange(10),minor=True)

			# ax2.set_yscale('log')
			ax2.set_xlim(0,2.86)
			ax2.tick_params(axis='both', which='both', color='white')
			if i==1: ax2.set_ylabel(r'$M_\ast$ [$10^7$ M$_\odot$]')
			axlist = np.append(axlist,ax2)

		print('-----------------------------------------------------------------\n')

	#---finish-setting-up-plot---------------------------------------------------------------------
	for ax in axlist:
		ax.spines['bottom'].set_color(axcolor)
		ax.spines['top'].set_color(axcolor)
		ax.spines['left'].set_color(axcolor)
		ax.spines['right'].set_color(axcolor)
		ax.tick_params(axis='both', which='both', color=axcolor)

	do_horiz_cbar = True

	if do_horiz_cbar:
		if len(models)==3: yanch = 10.0
		elif len(models)==7: yanch = 8.0
		cbar = fig.colorbar(c, anchor=(0.0,yanch), orientation='horizontal', ax=axlist.ravel().tolist(),
			label=r'$M_\mathregular{star}$ / $M_\mathregular{DM}$')
		cbar.ax.xaxis.set_ticks_position('top')
		cbar.ax.xaxis.set_label_position('top')

	else:
		cbar = fig.colorbar(c, ax=axlist.ravel().tolist(),
			ticks=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5],label=r'log$_{10}$(SFR [M$_\odot$ yr$^{-1}$])')

	axarr[6].set_xlim(0,2.86)
	axarr[6].set_xlabel('time [Gyr]')
	axarr[6].set_xticks([0,0.5,1,1.5,2,2.5])
	axarr[6].set_xticks(np.arange(0,2.86,0.1),minor=True)	
	axarr[6].set_xticklabels(['0','0.5','1','1.5','2','2.5'])

	yminticks = np.append(np.arange(0.1,1,0.1),np.arange(1,3,1))
	# axarr[0].set_ylabel('distance [kpc]')
	axarr[0].set_yscale('log')
	axarr[0].set_ylim(0.1,5)
	axarr[0].set_yticks([0.1,0.2,0.5,1,2,5])
	axarr[0].set_yticklabels(['0.1','0.2','0.5','1','2','5'])
	axarr[0].set_yticks(yminticks,minor=True)

	axarr[1].set_ylabel('distance [kpc]')
	for j in np.arange(1,len(models)):
		axarr[j].set_yscale('log')
		axarr[j].set_ylim(0.1,5)
		axarr[j].set_yticks([0.1,0.2,0.5,1,2])
		axarr[j].set_yticklabels(['0.1','0.2','0.5','1','2'])
		axarr[j].set_yticks(yminticks,minor=True)


	# cbar.ax.set_yticks[]
	p.finalize(fig,'starfrac_color_'+whichsims,save=save,tight=True)
	# if save:
	# 	plt.savefig(d.plotdir+month+'/SFRcolor_time_radius_panels_'+whichsims+'.png',format='png',dpi=200)
	# else:
	# 	plt.show()

def type4frac_color_panels(do_r50=True,do_core=True,do_Mstar=False):
	print('plotting time_radius_sfrcolor_panels')
	
	# if not(whichsims=='ff_eSF_var_1e6' or whichsims=='eSF_eSFramp_eSF2_1e6'): 
	# 	raise ValueError('please use whichsims = ff_eSF_var_1e6 or eSF_eSFramp_eSF2_1e6')

	if len(models)==3:
		fig,axarr = p.makefig('3_vert',figx=10,figy=10)
	elif len(models)==7:
		fig,axarr = p.makefig('7_vert',figx=10,figy=20.5)
	else:
		raise ValueError('len of models is wrong. needs to be 3 or 7')

	dist_bins = np.logspace(-2,np.log10(5),100)
	
	fig,axarr = p.makefig('3_vert',figx=10,figy=10)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)

	axcolor = 'white'
	axlist = axarr

	an_x_pos = 0.35

	if do_Mstar:
		text = axarr[0].annotate(r'$M_\ast$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='orange',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_core:
		text = axarr[0].annotate(r'$r_\mathregular{core}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='turquoise',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_r50:
		text = axarr[0].annotate(r'$r_\mathregular{50, SF gas}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='white',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		

	for i,sim in enumerate(models):
		print(sim)
		text = axarr[i].annotate(models_label[i],xy=(0.05,0.8),xycoords='axes fraction',color='white',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='white', edgecolor='black'), path_effects.Normal()])

		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		try:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		except:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')

		drange = np.array(f['drange'])
		star_profile_all = np.array(f['type4'])
		dark_profile_all = np.array(f['dark'])
		f.close()
		
		max_snap = dark_profile_all.shape[0] - 1
		print('max_snap = '+str(max_snap))

		sel = (drange > 0.1) & (drange < 5)
		star_profile_all = star_profile_all[:,sel]
		dark_profile_all = dark_profile_all[:,sel]
		dist_bins = drange[sel]
		
		frac_profile_all = star_profile_all / dark_profile_all

		print('plotting gasfrac color')
		print('max frac: '+str(np.amax(frac_profile_all)))
		print('min frac: '+str(np.amin(frac_profile_all[frac_profile_all > 0])))

		if max_snap < 400:
			new_zs = np.zeros((400 - max_snap, len(dist_bins))) - 5
			frac_profile_all = np.vstack((frac_profile_all,new_zs))

		x,y = np.meshgrid(time,dist_bins)
		z = np.log10(frac_profile_all.T)
		# z = frac_profile_all.T
		c = axarr[i].pcolormesh(x, y, z, cmap='inferno', vmin=-1.5, vmax=1)

		#---plot-horizontal-lines------------------------------------------------------------------
		axarr[i].axhline(y=0.2,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=0.5,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=1,ls='--',color='white',lw=0.7,alpha=0.3)

		#---plot-core-radius-----------------------------------------------------------------------
		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		use_nfw_core = True

		if do_core:
			print('plotting core radius')
			if use_nfw_core:
				try: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3_hires.hdf5','r')
				except: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3.hdf5','r')
				
				core_radius = np.array(nf['core_radius'])
				nf.close()
				print('using NFW core')
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='black',lw=4)
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='turquoise',lw=1)
			else:
				try:
					nf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5','r')
				except:
			 		nf = h5py.File(d.datdir+'PITS_params_'+sim+'.hdf5','r')
				core_radius = np.array(nf['rc'])
				nf.close()
				print('using PITS core')

				dt = 0.05
				timebins = np.arange(0,np.amax(time)+dt,dt)
				binwidth = (timebins[1] - timebins[0])/2

				rcore_mean = np.array([])
				for j in range(len(timebins)-1):
					leftbin = timebins[j]
					rightbin = timebins[j+1]

					selbin = (time > leftbin) & (time < rightbin)
					rcore_mean = np.append(rcore_mean, np.mean(core_radius[selbin]))

				time_plt = timebins[0:-1]+binwidth
				axarr[i].plot(time_plt,rcore_mean,color='turquoise',lw=0.7)

				# axarr[i].plot(time,core_radius,'-',c='turquoise',lw=0.7)

		#---plot-r50SFgas--------------------------------------------------------------------------
		if do_r50:
			print('plotting r50')
			f = h5py.File(d.datdir+'r50SFgas_'+sim+'.hdf5','r')
			all_r50 = np.array(f['all_r50'])
			f.close()

			fn = d.datdir+'num_sfr_time_'+sim+'txt'
			try:
				print('read num_sfr from file')
				num_sfr = np.loadtxt(fn,dtype=float,delimiter=',')
			except:
				print('calculating num_sfr')
				num_sfr = np.array([])
				for snapnum in np.arange(0,max_snap+1):
					printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
					sys.stdout.write(printthing); sys.stdout.flush()
					if not(snapnum==max_snap+1): sys.stdout.write("\b" * (len(printthing)))
					else: print('')
					snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
					gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
					num_sfr = np.append(num_sfr, np.count_nonzero(gas_sfr > 0))
				np.savetxt(fn,num_sfr,delimiter=',')
				print('wrote num_sfr to '+fn)

			sel = num_sfr <= 10
			all_r50[sel] = np.nan

			time = time[:max_snap+1]

			dt = 0.025
			timebins = np.arange(0,np.amax(time)+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2

			r50_mean = np.array([])
			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				selbin = (time > leftbin) & (time < rightbin) & np.logical_not(np.isnan(all_r50))
				r50_mean = np.append(r50_mean, np.mean(all_r50[selbin]))

			time_plt = timebins[0:-1]+binwidth
			axarr[i].plot(time_plt,r50_mean,color='black',lw=2)
			axarr[i].plot(time_plt,r50_mean,color='white',lw=0.7)

		#---plot-cumulative-Mstar------------------------------------------------------------------
		if do_Mstar:
			print('plotting M_star')
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
			type4_profile_all = np.array(f['type4'])
			f.close()

			M_star_all = np.array([])
			for profile in type4_profile_all:
				M_star_all = np.append(M_star_all,profile[-1])

			M_star_all = M_star_all/(1.e7)

			ax2 = axarr[i].twinx()
			ax2.plot(time,M_star_all,c='orange',lw=1)
			ax2.set_ylim(0,10)

			if i==0: ax2.set_yticks([0,2,4,6,8,10])
			else: ax2.set_yticks([0,2,4,6,8])
			ax2.set_yticks(np.arange(10),minor=True)

			# ax2.set_yscale('log')
			ax2.set_xlim(0,2.86)
			ax2.tick_params(axis='both', which='both', color='white')
			if i==1: ax2.set_ylabel(r'$M_\ast$ [$10^7$ M$_\odot$]')
			axlist = np.append(axlist,ax2)

		print('-----------------------------------------------------------------\n')

	#---finish-setting-up-plot---------------------------------------------------------------------
	for ax in axlist:
		ax.spines['bottom'].set_color(axcolor)
		ax.spines['top'].set_color(axcolor)
		ax.spines['left'].set_color(axcolor)
		ax.spines['right'].set_color(axcolor)
		ax.tick_params(axis='both', which='both', color=axcolor)

	do_horiz_cbar = True

	if do_horiz_cbar:
		if len(models)==3: yanch = 10.0
		elif len(models)==7: yanch = 8.0
		cbar = fig.colorbar(c, anchor=(0.0,yanch), orientation='horizontal', ax=axlist.ravel().tolist(),
			ticks=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5],label=r'log$_{10}$(SFR [M$_\odot$ yr$^{-1}$])')
		cbar.ax.xaxis.set_ticks_position('top')
		cbar.ax.xaxis.set_label_position('top')

	else:
		cbar = fig.colorbar(c, ax=axlist.ravel().tolist(),
			ticks=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5],label=r'log$_{10}$(SFR [M$_\odot$ yr$^{-1}$])')

	axarr[6].set_xlim(0,2.86)
	axarr[6].set_xlabel('time [Gyr]')
	axarr[6].set_xticks([0,0.5,1,1.5,2,2.5])
	axarr[6].set_xticks(np.arange(0,2.86,0.1),minor=True)	
	axarr[6].set_xticklabels(['0','0.5','1','1.5','2','2.5'])

	yminticks = np.append(np.arange(0.1,1,0.1),np.arange(1,3,1))
	# axarr[0].set_ylabel('distance [kpc]')
	axarr[0].set_yscale('log')
	axarr[0].set_ylim(0.1,5)
	axarr[0].set_yticks([0.1,0.2,0.5,1,2,5])
	axarr[0].set_yticklabels(['0.1','0.2','0.5','1','2','5'])
	axarr[0].set_yticks(yminticks,minor=True)

	axarr[1].set_ylabel('distance [kpc]')
	for j in np.arange(1,len(models)):
		axarr[j].set_yscale('log')
		axarr[j].set_ylim(0.1,5)
		axarr[j].set_yticks([0.1,0.2,0.5,1,2])
		axarr[j].set_yticklabels(['0.1','0.2','0.5','1','2'])
		axarr[j].set_yticks(yminticks,minor=True)


	# cbar.ax.set_yticks[]
	p.finalize(fig,'type4frac_color_'+whichsims,save=save,tight=True)
	# if save:
	# 	plt.savefig(d.plotdir+month+'/SFRcolor_time_radius_panels_'+whichsims+'.png',format='png',dpi=200)
	# else:
	# 	plt.show()

def baryfrac_color_panels(do_r50=True,do_core=True,do_Mstar=False):
	print('plotting time_radius_sfrcolor_panels')

	if len(models)==3:
		fig,axarr = p.makefig('3_vert',figx=10,figy=10)
	elif len(models)==7:
		fig,axarr = p.makefig('7_vert',figx=10,figy=20.5)
	else:
		raise ValueError('len of models is wrong. needs to be 3 or 7')

	dist_bins = np.logspace(-2,np.log10(5),100)
	
	fig,axarr = p.makefig('3_vert',figx=10,figy=10)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)

	axcolor = 'white'
	axlist = axarr

	an_x_pos = 0.35

	if do_Mstar:
		text = axarr[0].annotate(r'$M_\ast$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='orange',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_core:
		text = axarr[0].annotate(r'$r_\mathregular{core}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='turquoise',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_r50:
		text = axarr[0].annotate(r'$r_\mathregular{50, SF gas}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='white',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		

	for i,sim in enumerate(models):
		print(sim)
		text = axarr[i].annotate(models_label[i],xy=(0.05,0.8),xycoords='axes fraction',color='white',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='white', edgecolor='black'), path_effects.Normal()])

		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		try:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
		except:
			f = h5py.File(d.datdir+'massprofiles_'+sim+'.hdf5','r')

		drange = np.array(f['drange'])
		bary_profile_all = np.array(f['type4']) + np.array(f['type3']) + np.array(f['type2']) + np.array(f['gas'])
		dark_profile_all = np.array(f['dark'])
		f.close()
		
		max_snap = dark_profile_all.shape[0] - 1
		print('max_snap = '+str(max_snap))

		sel = (drange > 0.1) & (drange < 5)
		bary_profile_all = bary_profile_all[:,sel]
		dark_profile_all = dark_profile_all[:,sel]
		dist_bins = drange[sel]
		
		frac_profile_all = bary_profile_all / dark_profile_all

		print('plotting baryfrac color')
		print('max frac: '+str(np.amax(frac_profile_all)))
		print('min frac: '+str(np.amin(frac_profile_all[frac_profile_all > 0])))

		if max_snap < 400:
			new_zs = np.zeros((400 - max_snap, len(dist_bins))) - 5
			frac_profile_all = np.vstack((frac_profile_all,new_zs))

		x,y = np.meshgrid(time,dist_bins)
		# z = np.log10(frac_profile_all.T)
		z = frac_profile_all.T
		c = axarr[i].pcolormesh(x, y, z, cmap='inferno', vmin=0, vmax=1)

		#---plot-horizontal-lines------------------------------------------------------------------
		axarr[i].axhline(y=0.2,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=0.5,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=1,ls='--',color='white',lw=0.7,alpha=0.3)

		#---plot-core-radius-----------------------------------------------------------------------
		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		use_nfw_core = True

		if do_core:
			print('plotting core radius')
			if use_nfw_core:
				try: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3_hires.hdf5','r')
				except: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3.hdf5','r')
				
				core_radius = np.array(nf['core_radius'])
				nf.close()
				print('using NFW core')
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='black',lw=4)
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='turquoise',lw=1)
			else:
				try:
					nf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5','r')
				except:
			 		nf = h5py.File(d.datdir+'PITS_params_'+sim+'.hdf5','r')
				core_radius = np.array(nf['rc'])
				nf.close()
				print('using PITS core')

				dt = 0.05
				timebins = np.arange(0,np.amax(time)+dt,dt)
				binwidth = (timebins[1] - timebins[0])/2

				rcore_mean = np.array([])
				for j in range(len(timebins)-1):
					leftbin = timebins[j]
					rightbin = timebins[j+1]

					selbin = (time > leftbin) & (time < rightbin)
					rcore_mean = np.append(rcore_mean, np.mean(core_radius[selbin]))

				time_plt = timebins[0:-1]+binwidth
				axarr[i].plot(time_plt,rcore_mean,color='turquoise',lw=0.7)

				# axarr[i].plot(time,core_radius,'-',c='turquoise',lw=0.7)

		#---plot-r50SFgas--------------------------------------------------------------------------
		if do_r50:
			print('plotting r50')
			f = h5py.File(d.datdir+'r50SFgas_'+sim+'.hdf5','r')
			all_r50 = np.array(f['all_r50'])
			f.close()

			fn = d.datdir+'num_sfr_time_'+sim+'txt'
			try:
				print('read num_sfr from file')
				num_sfr = np.loadtxt(fn,dtype=float,delimiter=',')
			except:
				print('calculating num_sfr')
				num_sfr = np.array([])
				for snapnum in np.arange(0,max_snap+1):
					printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
					sys.stdout.write(printthing); sys.stdout.flush()
					if not(snapnum==max_snap+1): sys.stdout.write("\b" * (len(printthing)))
					else: print('')
					snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
					gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
					num_sfr = np.append(num_sfr, np.count_nonzero(gas_sfr > 0))
				np.savetxt(fn,num_sfr,delimiter=',')
				print('wrote num_sfr to '+fn)

			sel = num_sfr <= 10
			all_r50[sel] = np.nan

			time = time[:max_snap+1]

			dt = 0.025
			timebins = np.arange(0,np.amax(time)+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2

			r50_mean = np.array([])
			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				selbin = (time > leftbin) & (time < rightbin) & np.logical_not(np.isnan(all_r50))
				r50_mean = np.append(r50_mean, np.mean(all_r50[selbin]))

			time_plt = timebins[0:-1]+binwidth
			axarr[i].plot(time_plt,r50_mean,color='black',lw=2)
			axarr[i].plot(time_plt,r50_mean,color='white',lw=0.7)

		#---plot-cumulative-Mstar------------------------------------------------------------------
		if do_Mstar:
			print('plotting M_star')
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
			type4_profile_all = np.array(f['type4'])
			f.close()

			M_star_all = np.array([])
			for profile in type4_profile_all:
				M_star_all = np.append(M_star_all,profile[-1])

			M_star_all = M_star_all/(1.e7)

			ax2 = axarr[i].twinx()
			ax2.plot(time,M_star_all,c='orange',lw=1)
			ax2.set_ylim(0,10)

			if i==0: ax2.set_yticks([0,2,4,6,8,10])
			else: ax2.set_yticks([0,2,4,6,8])
			ax2.set_yticks(np.arange(10),minor=True)

			# ax2.set_yscale('log')
			ax2.set_xlim(0,2.86)
			ax2.tick_params(axis='both', which='both', color='white')
			if i==1: ax2.set_ylabel(r'$M_\ast$ [$10^7$ M$_\odot$]')
			axlist = np.append(axlist,ax2)

		print('-----------------------------------------------------------------\n')

	#---finish-setting-up-plot---------------------------------------------------------------------
	for ax in axlist:
		ax.spines['bottom'].set_color(axcolor)
		ax.spines['top'].set_color(axcolor)
		ax.spines['left'].set_color(axcolor)
		ax.spines['right'].set_color(axcolor)
		ax.tick_params(axis='both', which='both', color=axcolor)

	do_horiz_cbar = True

	if do_horiz_cbar:
		if len(models)==3: yanch = 10.0
		elif len(models)==7: yanch = 8.0
		cbar = fig.colorbar(c, anchor=(0.0,yanch), orientation='horizontal', ax=axlist.ravel().tolist(),
			ticks=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5],label=r'log$_{10}$(SFR [M$_\odot$ yr$^{-1}$])')
		cbar.ax.xaxis.set_ticks_position('top')
		cbar.ax.xaxis.set_label_position('top')

	else:
		cbar = fig.colorbar(c, ax=axlist.ravel().tolist(),
			ticks=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5],label=r'log$_{10}$(SFR [M$_\odot$ yr$^{-1}$])')

	axarr[6].set_xlim(0,2.86)
	axarr[6].set_xlabel('time [Gyr]')
	axarr[6].set_xticks([0,0.5,1,1.5,2,2.5])
	axarr[6].set_xticks(np.arange(0,2.86,0.1),minor=True)	
	axarr[6].set_xticklabels(['0','0.5','1','1.5','2','2.5'])

	yminticks = np.append(np.arange(0.1,1,0.1),np.arange(1,3,1))
	# axarr[0].set_ylabel('distance [kpc]')
	axarr[0].set_yscale('log')
	axarr[0].set_ylim(0.1,5)
	axarr[0].set_yticks([0.1,0.2,0.5,1,2,5])
	axarr[0].set_yticklabels(['0.1','0.2','0.5','1','2','5'])
	axarr[0].set_yticks(yminticks,minor=True)

	axarr[1].set_ylabel('distance [kpc]')
	for j in np.arange(1,len(models)):
		axarr[j].set_yscale('log')
		axarr[j].set_ylim(0.1,5)
		axarr[j].set_yticks([0.1,0.2,0.5,1,2])
		axarr[j].set_yticklabels(['0.1','0.2','0.5','1','2'])
		axarr[j].set_yticks(yminticks,minor=True)


	# cbar.ax.set_yticks[]
	p.finalize(fig,'baryfrac_color_'+whichsims,save=save,tight=True)
	# if save:
	# 	plt.savefig(d.plotdir+month+'/SFRcolor_time_radius_panels_'+whichsims+'.png',format='png',dpi=200)
	# else:
	# 	plt.show()

def rvrcolor_panels(ptype,write=True,do_r50=True,do_core=True,do_Mstar=False):
	print('plotting time_radius_sfrcolor_panels')
	if not(ptype in ['gas','dark','star']):
		raise ValueError('please choose ptype = gas, dark, or star')

	if len(models)==3:
		fig,axarr = p.makefig('3_vert',figx=10,figy=10)
	elif len(models)==5:
		fig,axarr = p.makefig('5_vert',figx=10,figy=16.5)
	elif len(models)==7:
		fig,axarr = p.makefig('7_vert',figx=10,figy=20.5)
	else:
		raise ValueError('len of models is wrong. needs to be 3 or 7')

	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})
	# fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)

	dist_bins = np.logspace(-1,np.log10(5),50)

	axcolor = 'white'
	axlist = axarr

	an_x_pos = 0.35

	if do_Mstar:
		text = axarr[0].annotate(r'$M_\ast$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='orange',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_core:
		text = axarr[0].annotate(r'$r_\mathregular{core}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='turquoise',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		an_x_pos += 0.15
	if do_r50:
		text = axarr[0].annotate(r'$r_\mathregular{50, SF gas}$',xy=(an_x_pos,0.1),xycoords='axes fraction',color='white',size=18)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=2, facecolor='white', edgecolor='black'), path_effects.Normal()])
		
	if ptype=='gas': parttype = 0
	elif ptype=='dark': parttype = 1
	elif ptype=='star': parttype = 4

	for i,sim in enumerate(models):
		print(sim)
		text = axarr[i].annotate(models_label[i],xy=(0.05,0.8),xycoords='axes fraction',color='white',size=15)
		text.set_path_effects([path_effects.PathPatchEffect(linewidth=4, facecolor='white', edgecolor='black'), path_effects.Normal()])
		
		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		mypath = d.smuggledir+sim
		onlyfiles = np.array([f for f in listdir(mypath) if isfile(join(mypath, f))])
		a = np.sort(onlyfiles[np.flatnonzero(np.core.defchararray.find(onlyfiles,'snapshot')!=-1)])
		max_snap = int(a[-1].split('.')[0][-3:])
		print('max_snap = '+str(max_snap))

		cm_file = h5py.File(d.datdir+'centerofmass_'+sim+'.hdf5','r')
		CoM_all = np.array(cm_file['CoM'])
		cm_file.close()

		v_cm_file = h5py.File(d.datdir+'centerofmassvelocity_'+sim+'.hdf5','r')
		v_cm_all = np.array(v_cm_file['Vel_CoM'])
		v_cm_file.close()

		fname = d.datdir+'/'+ptype+'_vr_profile_all_'+sim+'.txt'

		#---plot-SFR-profile-colors----------------------------------------------------------------
		try:
			rvr_profile_all = np.loadtxt(fname,dtype=float,delimiter=',')
			print('successfully read sfr profiles data from file')
			preexist = True
		except:
			print('could not read data from file. calculating vr profiles manually ')
			preexist = False

			for snapnum in np.arange(0,max_snap+1):
				printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
				sys.stdout.write(printthing); sys.stdout.flush()
				if not(snapnum==max_snap): sys.stdout.write("\b" * (len(printthing)))
				else: print('')

				snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
				pos = snapHDF5.read_block(snapfile, 'POS ', parttype=parttype)/h - CoM_all[snapnum]
				vel = snapHDF5.read_block(snapfile, 'VEL ', parttype=parttype) - v_cm_all[snapnum]				

				sel_in_bound = (np.abs(pos[:,0]) < 5) & (np.abs(pos[:,1]) < 5) & (np.abs(pos[:,2]) < 5) & (np.abs(pos[:,0]) >= 0.1) & (np.abs(pos[:,1]) >= 0.1) & (np.abs(pos[:,2]) >= 0.1)
				pos = pos[sel_in_bound]
				vel = vel[sel_in_bound]

				dist = np.linalg.norm(pos,axis=1)

				vr_profile = np.array([])
				for j in np.arange(len(dist_bins)-1):
					seld = (dist >= dist_bins[j]) & (dist < dist_bins[j+1])

					if np.count_nonzero(seld) > 5:
						pos_this = pos[seld]
						dist_this = dist[seld]
						vel_this = vel[seld]
						
						r_hat_this = (pos_this.T / dist_this).T
						vr_this = np.diag(np.dot(r_hat_this,vel_this.T))

						vr_profile = np.append(vr_profile, np.median(vr_this))
					else:
						vr_profile = np.append(vr_profile, np.nan)

				# sel = np.isnan(vr_profile) & np.isinf(vr_profile)
				# vr_profile[sel] = 0

				if snapnum == 0:
					vr_profile_all = vr_profile
				else:
					vr_profile_all = np.vstack((vr_profile_all,vr_profile))
		if write:
			print('saving vr_profile_all to '+fname)
			np.savetxt(fname,vr_profile_all,delimiter=',')

		print('plotting Vr color')
		x,y = np.meshgrid(time,dist_bins)

		if max_snap < 400:
			new_zs = np.zeros((400 - max_snap, len(dist_bins) - 1)) - 5
			vr_profile_all = np.vstack((vr_profile_all,new_zs))

		z = vr_profile_all.T
		c = axarr[i].pcolormesh(x, y, z, cmap='inferno', vmin=-100, vmax=100)

		#---plot-horizontal-lines------------------------------------------------------------------
		axarr[i].axhline(y=0.2,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=0.5,ls='--',color='white',lw=0.7,alpha=0.3)
		axarr[i].axhline(y=1,ls='--',color='white',lw=0.7,alpha=0.3)

		#---plot-core-radius-----------------------------------------------------------------------
		tf = h5py.File(d.datdir+'timefile.hdf5','r')
		time = np.array(tf['time'])
		tf.close()

		use_nfw_core = True

		if do_core:
			print('plotting core radius')
			if use_nfw_core:
				try: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3_hires.hdf5','r')
				except: 
					nf = h5py.File(d.datdir+'coreradius_'+sim+'_dcut3.hdf5','r')
				
				core_radius = np.array(nf['core_radius'])
				nf.close()
				print('using NFW core')
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='black',lw=4)
				axarr[i].plot(time[:len(core_radius)],core_radius,'-',c='turquoise',lw=1)
			else:
				try:
					nf = h5py.File(d.datdir+'PITS_params_'+sim+'_hires.hdf5','r')
				except:
			 		nf = h5py.File(d.datdir+'PITS_params_'+sim+'.hdf5','r')
				core_radius = np.array(nf['rc'])
				nf.close()
				print('using PITS core')

				dt = 0.05
				timebins = np.arange(0,np.amax(time)+dt,dt)
				binwidth = (timebins[1] - timebins[0])/2

				rcore_mean = np.array([])
				for j in range(len(timebins)-1):
					leftbin = timebins[j]
					rightbin = timebins[j+1]

					selbin = (time > leftbin) & (time < rightbin)
					rcore_mean = np.append(rcore_mean, np.mean(core_radius[selbin]))

				time_plt = timebins[0:-1]+binwidth
				axarr[i].plot(time_plt,rcore_mean,color='turquoise',lw=0.7)

				# axarr[i].plot(time,core_radius,'-',c='turquoise',lw=0.7)

		#---plot-r50SFgas--------------------------------------------------------------------------
		if do_r50:
			print('plotting r50')
			f = h5py.File(d.datdir+'r50SFgas_'+sim+'.hdf5','r')
			all_r50 = np.array(f['all_r50'])
			f.close()

			fn = d.datdir+'num_sfr_time_'+sim+'txt'
			try:
				print('read num_sfr from file')
				num_sfr = np.loadtxt(fn,dtype=float,delimiter=',')
			except:
				print('calculating num_sfr')
				num_sfr = np.array([])
				for snapnum in np.arange(0,max_snap+1):
					printthing = 'calculating snapshot '+str(snapnum).zfill(3)+'/'+str(max_snap).zfill(3)
					sys.stdout.write(printthing); sys.stdout.flush()
					if not(snapnum==max_snap+1): sys.stdout.write("\b" * (len(printthing)))
					else: print('')
					snapfile = d.smuggledir+sim+'/snapshot_'+str(snapnum).zfill(3)
					gas_sfr = snapHDF5.read_block(snapfile, 'SFR ', parttype=0)
					num_sfr = np.append(num_sfr, np.count_nonzero(gas_sfr > 0))
				np.savetxt(fn,num_sfr,delimiter=',')
				print('wrote num_sfr to '+fn)

			sel = num_sfr <= 10
			all_r50[sel] = np.nan

			time = time[:max_snap+1]

			dt = 0.025
			timebins = np.arange(0,np.amax(time)+dt,dt)
			binwidth = (timebins[1] - timebins[0])/2

			r50_mean = np.array([])
			for j in range(len(timebins)-1):
				leftbin = timebins[j]
				rightbin = timebins[j+1]

				selbin = (time > leftbin) & (time < rightbin) & np.logical_not(np.isnan(all_r50))
				r50_mean = np.append(r50_mean, np.mean(all_r50[selbin]))

			time_plt = timebins[0:-1]+binwidth
			axarr[i].plot(time_plt,r50_mean,color='black',lw=2)
			axarr[i].plot(time_plt,r50_mean,color='white',lw=0.7)

		#---plot-cumulative-Mstar------------------------------------------------------------------
		if do_Mstar:
			print('plotting M_star')
			f = h5py.File(d.datdir+'massprofiles_'+sim+'_hires.hdf5','r')
			type4_profile_all = np.array(f['type4'])
			f.close()

			M_star_all = np.array([])
			for profile in type4_profile_all:
				M_star_all = np.append(M_star_all,profile[-1])

			M_star_all = M_star_all/(1.e7)

			ax2 = axarr[i].twinx()
			ax2.plot(time,M_star_all,c='orange',lw=1)
			ax2.set_ylim(0,10)

			if i==0: ax2.set_yticks([0,2,4,6,8,10])
			else: ax2.set_yticks([0,2,4,6,8])
			ax2.set_yticks(np.arange(10),minor=True)

			# ax2.set_yscale('log')
			ax2.set_xlim(0,2.86)
			ax2.tick_params(axis='both', which='both', color='white')
			if i==1: ax2.set_ylabel(r'$M_\ast$ [$10^7$ M$_\odot$]')
			axlist = np.append(axlist,ax2)

		print('-----------------------------------------------------------------\n')

	#---finish-setting-up-plot---------------------------------------------------------------------
	for ax in axlist:
		ax.spines['bottom'].set_color(axcolor)
		ax.spines['top'].set_color(axcolor)
		ax.spines['left'].set_color(axcolor)
		ax.spines['right'].set_color(axcolor)
		ax.tick_params(axis='both', which='both', color=axcolor)

	do_horiz_cbar = True

	if do_horiz_cbar:
		if len(models)==3: yanch = 10.0
		elif len(models)==7: yanch = 8.0
		cbar = fig.colorbar(c, anchor=(0.0,yanch), orientation='horizontal', ax=axlist.ravel().tolist()
			,label=r'$v_r$ [km s$^{-1}$]')
		cbar.ax.xaxis.set_ticks_position('top')
		cbar.ax.xaxis.set_label_position('top')

	else:
		cbar = fig.colorbar(c, ax=axlist.ravel().tolist(),
			ticks=[-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5],label=r'log$_{10}$(SFR [M$_\odot$ yr$^{-1}$])')

	axarr[6].set_xlim(0,2.86)
	axarr[6].set_xlabel('time [Gyr]')
	axarr[6].set_xticks([0,0.5,1,1.5,2,2.5])
	axarr[6].set_xticks(np.arange(0,2.86,0.1),minor=True)	
	axarr[6].set_xticklabels(['0','0.5','1','1.5','2','2.5'])

	yminticks = np.append(np.arange(0.1,1,0.1),np.arange(1,3,1))
	# axarr[0].set_ylabel('distance [kpc]')
	axarr[0].set_yscale('log')
	axarr[0].set_ylim(0.1,5)
	axarr[0].set_yticks([0.1,0.2,0.5,1,2,5])
	axarr[0].set_yticklabels(['0.1','0.2','0.5','1','2','5'])
	axarr[0].set_yticks(yminticks,minor=True)

	axarr[1].set_ylabel('distance [kpc]')
	for j in np.arange(1,len(models)):
		axarr[j].set_yscale('log')
		axarr[j].set_ylim(0.1,5)
		axarr[j].set_yticks([0.1,0.2,0.5,1,2])
		axarr[j].set_yticklabels(['0.1','0.2','0.5','1','2'])
		axarr[j].set_yticks(yminticks,minor=True)

	# axarr[2].set_yscale('log')
	# axarr[2].set_ylim(0.1,5)
	# axarr[2].set_yticks([0.1,0.2,0.5,1,2])
	# axarr[2].set_yticklabels(['0.1','0.2','0.5','1','2'])
	# axarr[2].set_yticks(yminticks,minor=True)


	# cbar.ax.set_yticks[]
	p.finalize(fig,'vrcolor_'+whichsims,save=save,tight=True)
	# if save:
	# 	plt.savefig(d.plotdir+month+'/SFRcolor_time_radius_panels_'+whichsims+'.png',format='png',dpi=200)
	# else:
	# 	plt.show()

def test_vtan():
	cm_file = h5py.File(d.datdir+'centerofmass_fiducial_1e6.hdf5','r')
	CoM_all = np.array(cm_file['CoM'])
	cm_file.close()

	v_cm_file = h5py.File(d.datdir+'centerofmassvelocity_fiducial_1e6.hdf5','r')
	v_cm_all = np.array(v_cm_file['Vel_CoM'])
	v_cm_file.close()

	snapfile = d.smuggledir+'fiducial_1e6/snapshot_400'
	# gasmass = snapHDF5.read_block(snapfile, 'MASS', parttype=0)*(1.e10)/h
	pos_all = snapHDF5.read_block(snapfile, 'POS ', parttype=1)/h - CoM_all[400]
	vel_all = snapHDF5.read_block(snapfile, 'VEL ', parttype=1) - v_cm_all[400]
	IDs_all = snapHDF5.read_block(snapfile, 'ID  ', parttype=1)

	# iD = np.random.choice(IDs_all)
	iD = 12218480
	print('\nID: '+str(iD)+'\n')
	r = pos_all[iD]
	# r_mag = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
	r_mag = np.linalg.norm(r)
	r_hat = r / r_mag

	v = vel_all[iD]
	# v_mag = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
	# vr = r_hat[0]*v[0] + r_hat[1]*v[1] + r_hat[2]*v[2]
	v_mag = np.linalg.norm(v)
	vr = np.dot(r_hat, v)

	# v_tan_1 = v_mag - np.abs(vr)
	v_tan_1 = np.sqrt(np.abs(vr**2 -  v_mag**2))

	v_tan_vector = np.cross(r_hat, v)
	v_tan_2 = np.linalg.norm(v_tan_vector)

	print('vector position [x,y,z]')
	print(r)
	print('')

	print('r hat [x,y,z]')
	print(r_hat)
	print('')

	print('vector velocity [x,y,z]')
	print(v)
	print('')

	print('total velocity')
	print(v_mag)
	print('')

	print('radial velocity')
	print(vr)
	print('')

	print('v_tan from subtracting')
	print(v_tan_1)
	print('')

	print('v_tan from magnitude of radial cross product')
	print(v_tan_2)




# for sim in models:
# 	calc.convert_sfr(sim)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# mstar_radial(do_bins=1)
# test_CoM_motion('fiducial_1e6')

# rhalf_sfgas()

# masstime(ptype='star')
# masstime(ptype='gas')
# masstime(ptype='dark')
# masstime(ptype='gasfrac')
# masstime(ptype='baryfrac')

# four_rho_plots_ratio()
twopanel_cores(hires=True,doPITS=False,fixed_dist=False)
# rho_hist_timeavg(do_median=1,normed=0)
sfr_masstime(do_bins=1,do_label=1)
# sigma_all_timeavg()
# vphi_profile(do_cold=False)

# time_radius_sfrcolor_panels(write = True)

# sfrcolor_panels(do_r50=False)
# gasfrac_color_panels(do_r50=False)
# starfrac_color_panels(do_r50=False)
# type4frac_color_panels(do_r50=False)
# baryfrac_color_panels(do_r50=False)

# str_rvr('fiducial_1e6',0)

# rvrcolor_panels(ptype='dark')




# test_vtan()








#--------------------------------------------------------------------------------------------------
print('\n')
pdb.set_trace()

