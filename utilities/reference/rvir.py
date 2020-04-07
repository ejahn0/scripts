import numpy as np 
import h5py
import pdb


def rvir(mvir, a=1.0):
	barydir = '/home/ethan/research/data/catalogs/metaldiff/m11q_res880/'
	f = h5py.File(barydir+'halos_600.hdf5','r')
	omega_m = np.float64(f['cosmology:omega_matter'])
	omega_l = np.float64(f['cosmology:omega_lambda'])
	f_b = np.float(np.array(f['cosmology:baryon.fraction']))
	h = 0.70200
	H0 = h*100
	G = 6.67408e-11

	H2 = (H0**2)*( omega_m*(a**(-3.0)) + omega_l )  #Friedmann equation, H2 = H^2
	rho_c = ( 3*H2/(8*np.pi*G) )*(1.552e-11)		#critical density w unit correction
	rvir_calc = np.cbrt( (3*mvir)/(4*np.pi*200*rho_c) )

	return rvir_calc


print rvir(1.e11)

pdb.set_trace()
