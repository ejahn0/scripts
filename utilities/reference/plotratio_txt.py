import numpy as np
import h5py
import matplotlib.pyplot as plt
import pylab as pl
import pdb
import scipy.spatial.distance as scidist
h = 0.702000

# sat_distance = input('enter the satellite cutoff distance in kpc: ')
# folder = str(input('enter the folder (\'metaldiff\' or \'core\'): '))
# m12_cut = input('enter the cutoff for m12i in kpc (100 or 50): ')

folder = 'metaldiff'

frac = 6 #input('enter cutoff distance as fraction of rvir (3 or 6): ')

if frac == 3:
	# sat_distance = 100 #kpc, m11q
	sat_distance = 100 #kpc, m11q
	m12_cut = 100 #kpc, m12i
if frac == 6:
	sat_distance = 20
	m12_cut = 50	


# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# ---------------------READ BARYON STUFF------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

print 'open bary file'
f_bary    = h5py.File('/media/ethan/Watson/research/data/catalogs/metaldiff/halos_600.hdf5','r')
# f_bary    = h5py.File('/home/ethan/data/catalogs/metaldiff/halos_600.hdf5','r')
m_b = np.float(np.array(f_bary['cosmology:baryon.fraction']))
vmax_bary = np.array( f_bary['vel.circ.max']  )#np.sqrt(1 - m_b)
mvir_bary = np.array( f_bary['mass.vir']      )/h
dhbc      = np.array( f_bary['host.distance'] )/h
mstr_bary = np.array( f_bary['star.mass']     )/h
f_bary.close()

# mstr_bary = mstr_bary[np.logical_not(np.isnan(mstr_bary))]
for i in range(len(mstr_bary)):
	if np.isnan(mstr_bary[i]):
		mstr_bary[i] = 0.

dhst_bary = np.array([])

for i in range(len(dhbc)):
	d = np.sqrt(dhbc[i,0]**2. + dhbc[i,1]**2. + dhbc[i,2]**2.)
	dhst_bary = np.append(dhst_bary,d)

# print 'open lores file'
# f_lores	  = h5py.File('/home/ethan/data/'+folder+'/res7100/halos_600.hdf5','r')
# LRm_b = np.float(np.array(f_lores['cosmology:baryon.fraction']))
# LRvmax_bary = np.array([ f_lores['vel.circ.max'][:]  ])#*np.sqrt(1 - LRm_b)
# LRmvir_bary = np.array([ f_lores['mass.vir'][:]      ])*h
# LRdhst_bary = np.array([ f_lores['host.distance'][:] ])*h
# LRmstr_bary = np.array([ f_lores['star.mass'][:]     ])*h
# f_lores.close()

print 'selecting baryonic data'
select_sat_bary = ((dhst_bary <= sat_distance) & (dhst_bary > 0.) & (mvir_bary > 0.) & (vmax_bary > 1.))
select_luminous = ((dhst_bary <= sat_distance) & (dhst_bary > 0.) & (mvir_bary > 0.) & (vmax_bary > 1.) & (mstr_bary > 0.))

# print 'selecting lores data'
# LRselect_sat_bary = ((LRdhst_bary <= sat_distance) & (LRdhst_bary > 0.) & (LRmvir_bary > 0.) & (LRvmax_bary > 0.))

vmax_lum = vmax_bary[select_luminous]
vmax_lum_ord = np.sort(vmax_lum)[::-1]
N_lum = np.arange(len(vmax_lum_ord))

vmax_sat_bary = vmax_bary[select_sat_bary]
vmax_sat_bary_ord = np.sort(vmax_sat_bary)[::-1]
N_sat_bary = np.arange(len(vmax_sat_bary_ord)) 

# LRvmax_sat_bary = LRvmax_bary[LRselect_sat_bary]
# LRvmax_sat_bary_ord = np.sort(LRvmax_sat_bary)[::-1]
# LRN_sat_bary = np.arange(len(LRvmax_sat_bary_ord)) 


# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# ---------------READ DARK MATTER ONLY STUFF--------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

print 'opening dark matter file'
# f_dm =  open('/home/ethan/data/catalogs/dmo/datafiles/data_600.txt','r') 
# f_dm =  open('/home/ethan/Desktop/data_NEWcat_600-2.txt','r') 
f_dm =  open('/media/ethan/Watson/research/data/catalogs/dmo/data_NEWcat_600.txt','r') 
datadm = np.array([])
#read first line
l = np.array(np.float64(f_dm.readline().split(' ')))
datadm = np.append(datadm,l)
#loop through file
for line in f_dm:
	datadm = np.vstack((datadm, np.array(np.float64(line.split(' ')))))

# already did unit corrections in get_dmo
vmax_dm = datadm[:,0]
vmax_dm_norm = datadm[:,0]*np.sqrt(1 - m_b)
mvir_dm = datadm[:,1]
pos_dm = datadm[:,2:]

print 'find host halo'
host_id = np.where(mvir_dm == np.amax(mvir_dm))[0] 	#find id of highest mvir
host_pos = pos_dm[host_id,:]						#take the whole row (x,y,z)

print 'calculate distances to host'
dhst_dm = np.array([])
for i in range(pos_dm.shape[0]):
	dhst_dm = np.append(dhst_dm,scidist.euclidean(pos_dm[i],host_pos))

dhst_dm = dhst_dm*1000

print 'selecting dark matter data'
select_sat_dm = ((dhst_dm <= sat_distance) & (dhst_dm > 0.) & (mvir_dm > 0.) & (vmax_dm > 1.))
vmax_sat_dm = vmax_dm[select_sat_dm]
vmax_sat_dm_ord = np.sort(vmax_sat_dm)[::-1]
N_sat_dm = np.arange(len(vmax_sat_dm_ord))

vmax_sat_dm_norm = vmax_dm_norm[select_sat_dm]
vmax_sat_dm_ord_norm = np.sort(vmax_sat_dm_norm)[::-1]
N_sat_dm_norm = np.arange(len(vmax_sat_dm_ord_norm))

# --------------------------------------------------------------------------------------------------------------------------
# --------------------READ M12I RATIO---------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------



print 'reading m12 data'
# f_m12 = open('/home/ethan/data/m12i/m12i_'+str(m12_cut)+'kpc_data.txt','r')
f_m12 = open('/media/ethan/Watson/research/data/m12i/m12i_'+str(m12_cut)+'kpc_data.txt','r')
datam12 = np.array([])
l = np.array(np.float64(f_m12.readline().split(', ')))
datam12 = np.append(datam12,l)
for line in f_m12:
	datam12 = np.vstack((datam12, np.array(np.float64(line.split(', ')))))
vmax_m12 = datam12[:,0]
lvmax_m12 = np.log10(vmax_m12)
ratio_m12 = datam12[:,1]



# --------------------------------------------------------------------------------------------------------------------------
# -------------------CALCULATE RATIO----------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------

print 'calculating ratio'
lvmax_sat_dm_ord = np.log10(vmax_sat_dm_ord)
lvmax_sat_dm_ord_norm = np.log10(vmax_sat_dm_ord_norm)
lvmax_sat_bary_ord = np.log10(vmax_sat_bary_ord)
lvmax_lum_ord = np.log10(vmax_lum_ord)
# lLRvmax_sat_bary_ord = np.log10(LRvmax_sat_bary_ord)


# BIN FROM THE LEFT in log(vmax) space using baryon set
binmin = np.amin(lvmax_sat_bary_ord)
binmax = np.amax(lvmax_sat_bary_ord)
bins = np.linspace(binmin,binmax,31)
bins = bins[0:len(bins)-1] #pop the max value
binwidth = bins[1]-bins[0]

N_ratio = np.array([])

for i in range(len(bins)):
	#get bin edges
	mn = bins[i]		
	mx = mn+binwidth	

	#select points where the vmax's are in the desired range
	aux_bary = ((lvmax_sat_bary_ord>=mn) & (lvmax_sat_bary_ord<mx))
	aux_dm = ((lvmax_sat_dm_ord>=mn) & (lvmax_sat_dm_ord<mx))

	#select the corresponding N's (y-values)
	N_bary = N_sat_bary[aux_bary]
	N_dm = N_sat_dm[aux_dm]

	#take mean of each y-vals
	mean_bary = np.mean(N_bary)
	mean_dm = np.mean(N_dm)

	#find ratio for this bin
	binratio = mean_dm/mean_bary

	#append to list of ratios for all bins
	N_ratio = np.append(N_ratio,binratio)

#the plotted x-value should be the center of each bin
vmax_binned = bins + binwidth 

#------------------------------------------------------------------------------------------------------------------------------------
#---------------CALCULATE RATIO FOR NORMED DM----------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
# BIN FROM THE LEFT in log(vmax) space using baryon set
# binmin = np.amin(lvmax_sat_bary_ord)
# binmax = np.amax(lvmax_sat_bary_ord)
# bins = np.linspace(binmin,binmax,31)
# bins = bins[0:len(bins)-1] #pop the max value
# binwidth = bins[1]-bins[0]
print 'calculating normed ratio'
N_ratio_norm = np.array([])
for i in range(len(bins)):
	#get bin edges
	mn = bins[i]		
	mx = mn+binwidth	

	#select points where the vmax's are in the desired range
	aux_bary = ((lvmax_sat_bary_ord>=mn) & (lvmax_sat_bary_ord<mx))
	aux_dm = ((lvmax_sat_dm_ord_norm>=mn) & (lvmax_sat_dm_ord_norm<mx))

	#select the corresponding N's (y-values)
	N_bary = N_sat_bary[aux_bary]
	N_dm = N_sat_dm_norm[aux_dm]

	#take mean of each y-vals
	mean_bary = np.mean(N_bary)
	mean_dm = np.mean(N_dm)

	#find ratio for this bin
	binratio = mean_dm/mean_bary

	#append to list of ratios for all bins
	N_ratio_norm = np.append(N_ratio_norm,binratio)

#the plotted x-value should be the center of each bin
# vmax_binned = bins + binwidth 


#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#--------------------PLOT-ARE-BELOW--------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
print 'plotting'
plt.clf()
f, axarr = plt.subplots(2, sharex=True, gridspec_kw = {'height_ratios':[2.5, 1]}) 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# MAKE EVERYTHING LOG BEFORE CALCULATING!! 
logN_bary = np.log10(N_sat_bary)
logN_dm = np.log10(N_sat_dm)
logN_dm_norm = np.log10(N_sat_dm_norm)
logN_lum = np.log10(N_lum)
# logLRN_bary = np.log10(LRN_sat_bary)
# # replace any infinites with zeros
# aux = np.invert(np.isfinite(logN_bary))
# logN_bary[aux]=0
# aux = np.invert(np.isfinite(logN_dm))
# logN_dm[aux]=0

#plot regular Mass Functions


axarr[0].plot(lvmax_sat_dm_ord,logN_dm,'-.',markersize=1.0,alpha=1,label="raw DM", color='#919191')
axarr[0].plot(lvmax_sat_dm_ord_norm,logN_dm_norm,'-',markersize=1.0,alpha=0.9,label="DMO", color='black')
axarr[0].plot(lvmax_lum_ord,logN_lum,'-o',markersize=3.0,alpha=0.7,label="luminous", color='#d19b30')
# axarr[0].plot(lLRvmax_sat_bary_ord,logLRN_bary,':',markersize=1.0,alpha=1,label="m11q, low res", color='#a56866')
axarr[0].plot(lvmax_sat_bary_ord,logN_bary,'-',markersize=1.0,alpha=0.9,label="m11q,"+str(sat_distance)+'kpc', color='#447fdd')

axarr[0].legend(fontsize='7',loc=0)
axarr[0].grid(b=True, which='major', color='#cccccc', linestyle=':',linewidth='0.5')
axarr[0].minorticks_on()
axarr[0].tick_params(direction='out', length=4, width=0.5, colors='black')
axarr[0].set_title('substructure ratio\n z=0')

axarr[0].set_ylabel(r'N $>$ v$_{max}$')
axarr[0].set_yticks([0,0.5,1.0,1.5,2.0,2.5,3.0])
axarr[0].set_yticklabels([r'$10^0$','',r'$10^{1}$','',r'$10^2$','',r'$10^3$'])
axarr[0].set_ylim(-0.1,3.2)

axarr[0].set_xticks([0.,0.7,1.,1.3])
axarr[0].set_xticklabels(['','','','',''])
# axarr[0].set_xlim(0,1.35)


#plot ratio
axarr[1].plot(vmax_binned,N_ratio,'-.',markersize=1.0,alpha=1,color='#919191')
axarr[1].plot(vmax_binned,N_ratio_norm,'-',markersize=1.0,alpha=1,color='black')
axarr[1].plot(lvmax_m12,ratio_m12,':',markersize=1.0,alpha=1,label='m12i,'+str(m12_cut)+'kpc',color='#515151')

axarr[1].legend(fontsize='7',loc=0)
axarr[1].grid(b=True, which='major', color='#cccccc', linestyle='-',linewidth='0.5')
axarr[1].minorticks_on()
axarr[1].tick_params(direction='out', length=4, width=0.5, colors='black')

axarr[1].set_ylabel(r"N / N$_{m11q}$")
# axarr[1].set_xlim(0,1.35)
# axarr[1].set_ylim(0.0,3.0)
# axarr[1].set_yticks([0,0.5,1,1.5,2,2.5,3])
# axarr[1].set_yticklabels(['0','','1','','2','','3'])

axarr[1].set_xlabel(r"v$_{max}$ (km/s)")
axarr[1].set_xticks([0.,0.7,1.,1.3])#,1.6])
axarr[1].set_xticklabels(['1','5','10','20'])#'40'])


#save plot to desktop
pl.savefig('/home/ethan/Desktop/ratio_newdata_'+str(sat_distance)+'kpc.png',format='png',dpi=1000)


pdb.set_trace()