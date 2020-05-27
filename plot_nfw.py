import numpy as np
import matplotlib.pyplot as plt
import pdb, sys
import math_helper as m
import plot_helper as p
# import faulthandler

# faulthandler.enable()

# plt.rcParams['text.usetex'] = True

G = m.Gprime	# kpc * (km/s)^2 / Msun
H = 70./1000. 	# 70 km/s/Mpc / 1000 kpc/Mpc = 0.07 km/s/kpc
rho_crit = 3*H**2 / (8*np.pi*G) 

#---------------------------

def get_rho_NFW(c,r200,rho_0,h=0.7,r_array=np.logspace(-1,2,100)):
	delta_c = 200./3. * c**3 / (np.log(1+c) - c/(1+c)) 
	rs = r200 / c
	# rho_array = delta_c*rho_crit/( (r_array/rs) * (1 + r_array/rs)**2. )
	rho_array = rho_0/( (r_array/rs) * (1 + r_array/rs)**2. )
	return rho_array

def get_M200(r200):
	#
	return 200.*rho_crit*4./3.*np.pi*r200**3

def M_NFW_in_r(c,r200,r):
	gc = 1. / (np.log(1+c) - c/(1+c)) 
	M200 = get_M200(r200)
	rs = r200 / c
	return M200*gc*(np.log(1 + r/rs) - (r/rs)*(1 + r/rs)**-1)

def test_r200():
	r200 = input('type an r200 to test or -1 to quit: ')

	while r200 != '-1':
		try:
			r200 = float(r200)
			print('M200 = ',m.scinote(M200(r200)))
			r200 =  input('type another r200 or -1 to quit: ')

		except:
			r200 = input('please type a number: ')

def plot_NFW(c_arr,r200_arr):
	fig,ax = p.makefig(1)

	if type(c_arr)==int or type(c_arr)==float:
		c_arr = np.array([c_arr])


	if type(r200_arr)==int or type(r200_arr)==float:
		r200_arr = np.array([r200_arr])
	
	for i,c in enumerate(c_arr):
		for j,r200 in enumerate(r200_arr):
			r,rho = rho_NFW(c,r200)
			ax.plot(r,rho)

	ax.set_xscale('log')
	ax.set_yscale('log')

	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$\rho$')

	p.finalize(fig,'nfw_plot',save=0)

def get_rho_cNFW(c,r200,K,tsf,eta,Rhalf,r_array=np.logspace(-1,2,100)):
	rs = r200 / c
	tdyn = 2*np.pi*np.sqrt(rs**3 / (m.Gsim*M_NFW_in_r(c,r200,rs)))
	tdyn = tdyn / (3.154e16)
	n = np.tanh(K*tsf/tdyn)
	rc = eta*Rhalf
	fn = (np.tanh(r_array/rc))**n
	fn_1 = (np.tanh(r_array/rc))**(n-1)
	f2 = (np.tanh(r_array/rc))**2

	rho_NFW = get_rho_NFW(c,r200)
	M_NFW = M_NFW_in_r(c,r200,r_array)

	return fn*rho_NFW + (n*fn_1*(1-f2))/(4*np.pi*r_array**2*rc)*M_NFW

def plot_cNFW(c,r200,K,tsf,eta,Rhalf):
	fig,ax = p.makefig(1)
	r_array = np.logspace(-1,2,100)
	rho_cNFW = get_rho_cNFW(c,r200,K,tsf,eta,Rhalf)
	ax.plot(r_array,rho_cNFW)
	ax.set_xscale('log')
	ax.set_yscale('log')

	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$\rho$')

	p.finalize(fig,'cnfw_plot',save=0)

def plot_both(c,r200,tsf,eta,K,Rhalf):
	fig,ax = p.makefig(1)
	r_array = np.logspace(-1,2,100)
	rho_NFW = get_rho_NFW(c,r200)
	ax.plot(r_array,rho_NFW,label='NFW')

	for i,K in enumerate(np.linspace(0,1,10)):
		rho_cNFW = get_rho_cNFW(c,r200,K,tsf,eta,Rhalf)
		ax.plot(r_array,rho_cNFW,label=r'cNFW, $\kappa$='+str(np.round(K,1)))

	ax.legend(prop={'size':11},frameon=False)

	ax.set_xscale('log')
	ax.set_yscale('log')

	ax.set_xlabel(r'$r$')
	ax.set_ylabel(r'$\rho$')

	p.finalize(fig,'cnfw_plot',save=0)






		

#---------------------------

# test_r200()

# plot_nfw(15,60)

# plot_cNFW(c=15,r200=60,K=0.04,tsf=1,eta=1.75,Rhalf=2)
# plot_both(c=15,r200=60,tsf=1,eta=1.75,K=0.04,Rhalf=2)

# plot_data()



# test_fn()



# pdb.set_trace()