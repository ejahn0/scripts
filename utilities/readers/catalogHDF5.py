import h5py, os, sys
import numpy as np
import directories as d
import math_helper as m
hostname = os.uname()[1]

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
starkeys = np.array(['star.form.time.100',
	'star.form.time.50',
	'star.form.time.90',
	'star.form.time.95',
	'star.form.time.dif.68',
	'star.indices',
	'star.mass',
	'star.mass.neutral',
	'star.massfraction',
	'star.number',
	'star.position',
	'star.radius.50',
	'star.radius.90',
	'star.vel.circ.50',
	'star.vel.std',
	'star.vel.std.50',
	'star.velocity'])
halokeys = np.array(['accrete.rate',
	'accrete.rate.100Myr',
	'accrete.rate.tdyn',
	'am.phantom',
	'am.progenitor.main',
	'axis.b_div.by_a',
	'axis.c_div.by_a',
	'cosmology:baryon.fraction',
	'cosmology:hubble',
	'cosmology:n_s',
	'cosmology:omega_baryon',
	'cosmology:omega_curvature',
	'cosmology:omega_dm',
	'cosmology:omega_lambda',
	'cosmology:omega_matter',
	'cosmology:sigma_8',
	'cosmology:w',
	'descendant.snapshot',
	'host.distance',
	'host.index',
	'host.velocity',
	'host.velocity.rad',
	'host.velocity.tan',
	'id',
	'infall.first.mass',
	'infall.first.snapshot',
	'infall.first.vel.circ.max',
	'infall.mass',
	'infall.snapshot',
	'infall.vel.circ.max',
	'info:baryonic',
	'info:box.length',
	'info:box.length_div.by_h',
	'info:dark.particle.mass',
	'info:gas.particle.mass',
	'info:host.number',
	'major.merger.snapshot',
	'mass',
	'mass.180m',
	'mass.200c',
	'mass.200m',
	'mass.500c',
	'mass.bound',
	'mass.half.snapshot',
	'mass.lowres',
	'mass.peak',
	'mass.peak.snapshot',
	'mass.vir',
	'position',
	'position.offset',
	'progenitor.number',
	'radius',
	'scale.radius',
	'scale.radius.klypin',
	'snapshot:index',
	'snapshot:redshift',
	'snapshot:scalefactor',
	'snapshot:time',
	'snapshot:time.hubble',
	'snapshot:time.lookback',
	'spin.bullock',
	'spin.peebles',
	'tree.index',
	'vel.circ.max',
	'vel.circ.peak',
	'vel.std',
	'velocity',
	'velocity.offset'])
notarrays = np.array(['cosmology:baryon.fraction',
	'cosmology:hubble',
	'cosmology:n_s',
	'cosmology:omega_baryon',
	'cosmology:omega_curvature',
	'cosmology:omega_dm',
	'cosmology:omega_lambda',
	'cosmology:omega_matter',
	'cosmology:sigma_8',
	'cosmology:w',
	'info:baryonic',
	'info:box.length',
	'info:box.length_div.by_h',
	'info:dark.particle.mass',
	'info:gas.particle.mass',
	'info:host.number',
	'snapshot:index',
	'snapshot:redshift',
	'snapshot:scalefactor',
	'snapshot:time',
	'snapshot:time.hubble',
	'snapshot:time.lookback',
	])

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def read(sim,snapID,*flags):
	if not('res' in sim):
		raise ValueError('Include the resolution in sim string! e.g. "m11q_res880"') 

	stuff = ()
	for flag in flags:
		if flag in starkeys:
			ftype = '/star'
			if 'dmo' in sim:
				raise ValueError('Passed a star key to a DMO simulation. Check ya shit!')

		elif (flag in halokeys) or (flag == 'distance.1d'):
			ftype = '/halo'
		else:
			raise ValueError('Unknown Flag! :(')

		#if snapID is a string, it is the snapshot number
		if (type(snapID)==str) or (type(snapID)==np.string_) or (type(snapID)==np.str_):
			catalog = h5py.File(d.catdir+sim+ftype+'_'+snapID+'.hdf5','r')

		#if snapID is an int or float, it is the redshift
		elif (type(snapID)==float) or (type(snapID)==int) or (type(snapID)==np.float64) or (type(snapID)==np.int64):
			num = m.get_snapnum(snapID)
			catalog = h5py.File(d.catdir+sim+ftype+'_'+num+'.hdf5','r')
		else:
			raise ValueError('unknown "snapID" type')

		if flag in notarrays:
			data = np.float64(catalog[flag])
		elif flag=='host.index':
			data = np.array(catalog[flag])[0]
		elif flag=='distance.1d':
			hostIndex = np.array(catalog['host.index'])[0]
			dhost_3D = np.array(catalog['host.distance'])
			data = np.linalg.norm(dhost_3D-dhost_3D[hostIndex],axis=1)
		else:
			data = np.array(catalog[flag])
		catalog.close()

		stuff = stuff + (data,)

	if len(stuff) > 1:
		return stuff
	else:
		return stuff[0]

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------