import numpy as np
import matplotlib.pyplot as plt
import pdb, sys, h5py
import math_helper as m
import plot_helper as p

plt.rcParams['text.usetex'] = True

#---get-data-and-parameters------------------------------------------------------------------------
datdir = '/home/ethan/research/data/hdf5/'
f = h5py.File(datdir+'massprofiles_fiducial_1e5.hdf5','r')
drange = np.array(f['drange'])
vols = 4./3.*np.pi*(drange**3)
rho_DM = np.array(f['dark'])[400] / vols
f.close()

G = m.Gprime	# kpc * (km/s)^2 / Msun
H = 70./1000. 	# 70 km/s/Mpc / 1000 kpc/Mpc = 0.07 km/s/kpc
rho_crit = 3*H**2 / (8*np.pi*G) 
r200 = drange[(rho_DM >= 200*rho_crit)][-1]

rho_DM = rho_DM[(drange > 1)]
drange = drange[(drange > 1)]

c_true = 15
rho0_true = 5e7

#--------------------------------------------------------------------------------------------------
def NFW_model(pars,x):
	# pars = [c,rho0]
	c = pars[0]; rho_0 = pars[1]
	rs = r200 / c
	return rho_0 / ( (x/rs) * (1 + x/rs)**2. )
	 
class NFW_model_bayes(object):

	def __init__(self,x,y):
		self.x=np.asarray(x)
		self.y=np.asarray(y)

	def ln_likelihood(self, pars):
		N = len(self.y)
		dy = self.y - NFW_model(pars, self.x)
		# ivar = 1 / self.y_err**2 # inverse-variance
		# return -0.5 * (N*np.log(2*np.pi) + np.sum(2*np.log(self.y_err)) + np.sum(dy**2 * ivar))
		return -0.5 * (N*np.log(2*np.pi) + np.sum(dy**2))

	def ln_prior(self, pars):
		c, rho0 = pars # unpack parameters
		ln_prior_val = 0. # we'll add to this

		if c < 10 or c > 20:
			return -np.inf
		else:
			ln_prior_val += np.log(1e-2) # normalization, log(1/100)

		if rho0 < 1e5 or rho0 > 1e9:
			return -np.inf
		else:
			ln_prior_val += np.log(1e-2) # normalization, log(1/100)

		return ln_prior_val

	def ln_posterior(self, pars):
		lnp = self.ln_prior(pars)
		if np.isinf(lnp):
			return lnp

		lnL = self.ln_likelihood(pars)
		lnprob = lnp + lnL
		
		if np.isnan(lnprob):
			return -np.inf

		return lnprob

	def __call__(self, pars):
		return self.ln_posterior(pars)


def evaluate_on_grid(func, c_grid, rho0_grid, args=()):
	c_grid,rho0_grid = np.meshgrid(c_grid, rho0_grid)
	cr_grid = np.vstack((c_grid.ravel(), rho0_grid.ravel())).T
    
	func_vals = np.zeros(cr_grid.shape[0])
	for j,pars in enumerate(cr_grid):
		func_vals[j] = func(pars, *args)
        
	return func_vals.reshape(c_grid.shape)


def test():
	fig,axes = plt.subplots(1, 3, figsize=(14,5.1), sharex=True, sharey=True)

	model = NFW_model_bayes(drange,rho_DM)

	# make a 256x256 grid of parameter values centered on the true values
	c_grid = np.linspace(c_true-5., c_true+5, 256)
	rho0_grid = np.linspace(rho0_true*0.5, rho0_true*2, 256)

	ln_prior_vals = evaluate_on_grid(model.ln_prior, c_grid, rho0_grid)
	ln_like_vals = evaluate_on_grid(model.ln_likelihood, c_grid, rho0_grid)
	ln_post_vals = evaluate_on_grid(model.ln_posterior, c_grid, rho0_grid)

	for i,vals in enumerate([ln_prior_vals, ln_like_vals, ln_post_vals]):
		axes[i].pcolormesh(c_grid, rho0_grid, vals, 
			cmap='Blues', vmin=vals.max()*2, vmax=vals.max()) # arbitrary scale
	
	# print(vals.max()/2)
	# print(vals.max() )
	# print(vals.max())
	axes[0].set_title('log-prior', fontsize=20)
	axes[1].set_title('log-likelihood', fontsize=20)
	axes[2].set_title('log-posterior', fontsize=20)
	
	# print()


	for ax in axes:
		ax.set_xlabel('c')
	    
		# plot the truth
		ax.plot(c_true, rho0_true, marker='o', zorder=10, color='#de2d26')
		ax.axis('tight')
		ax.set_xlim(5,25)

	axes[0].set_ylabel(r'$\rho0$')
	axes[0].set_yscale('log')


	# fig.tight_layout()
	plt.show()

#--------------------------------------------------------------------------------------------------

test()


pdb.set_trace()