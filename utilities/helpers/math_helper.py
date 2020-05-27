import numpy as np
import directories as d
import catalogHDF5 as cat
import snapHDF5


#--------------------------------------------------------------------------------------------------
#---constants--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
G       = 6.67408e-11 # SI units : [N m^2 / kg^2] = [m^3 / s^2 kg]
Gprime  = 4.3257e-6   # units: kpc km^2 / s^2 Msun :: give mass in Msun and distance in kpc, get V in km/s
Mpc_km  = 3.241e-20   # Mpc / km   <- backwards
s_Gyr   = 3.171e-17   # Gyr / s    <- backwards
kg_Msun = 1.9884e30   # kg  / Msun <- backwards
m_kpc   = 3.086e19    #  m  / kpc
Gsim    = G*(kg_Msun)/(m_kpc**3) # units: kpc^3 / s^2 Msun :: to use with values taken directly from the snapshots

to_Gyr = Mpc_km / s_Gyr   # multiply by this to convert H0 from km/s/Mpc to 1/Gyr
to_Msun_kpc3 = (Mpc_km)**2 * (kg_Msun)**(-1) * (m_kpc)**3 # convert rho_crit to Msun / kpc^3

species = np.array(['total Z','He','C','N','O','Ne','Mg','Si','S','Ca','Fe'])
#                          total Z   He     C        N        O        Ne       Mg       Si       S        Ca       Fe
SolarAbundances = np.array([0.02, 0.2485, 3.26e-3, 1.32e-3, 8.65e-3, 2.22e-3, 9.31e-4, 1.08e-3, 6.44e-4, 1.01e-4, 1.73e-3])
m12sims = np.array(['m12b_res7100','m12c_res7100','m12f_res7100','m12i_res7100','m12m_res7100','m12r_res7100','m12w_res7100'])
m12minVmax = np.array([4.2,4.0,4.0,4.0,4.3,4.1,4.5])

#---from-smuggle-plots-----------------------------------------------------------------------------
UnitLength_in_cm =        3.085678e21 #  1.0 kpc
UnitMass_in_g    =        1.989e43 #  1.0e10 solar masses
UnitVelocity_in_cm_per_s = 1.e5  #  1 km/sec
UnitTime_in_s= UnitLength_in_cm / UnitVelocity_in_cm_per_s
UnitDensity_in_cgs= UnitMass_in_g/ UnitLength_in_cm**3
UnitPressure_in_cgs= UnitMass_in_g/ UnitLength_in_cm/ UnitTime_in_s**2
UnitEnergy_in_cgs= UnitMass_in_g * UnitLength_in_cm**2 / UnitTime_in_s**2
GRAVITY   = 6.672e-8
BOLTZMANN = 1.3806e-16
PROTONMASS = 1.6726e-24
Gsmug=GRAVITY/ UnitLength_in_cm**3 * UnitMass_in_g * UnitTime_in_s**2
Xh=0.76                        # mass fraction of hydrogen
h = 0.73
gamma= 5.0/3.


#--------------------------------------------------------------------------------------------------
#---functions--------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
def get_snapnum(z):
	snapnums = np.loadtxt(d.fileno_dir, dtype=float, delimiter=' ', usecols=[0])
	redshifts = np.loadtxt(d.fileno_dir, dtype=float, delimiter=' ', usecols=[1])
	return str(int(snapnums[np.where(redshifts == find_nearest(redshifts, z))[0][0]])).zfill(3)

def get_snapz(num):
	snapnums = np.loadtxt(d.fileno_dir, dtype=float, delimiter=' ', usecols=[0])
	redshifts = np.loadtxt(d.fileno_dir, dtype=float, delimiter=' ', usecols=[1])
	return redshifts[np.where(snapnums == num)[0][0]]

def find_nearest(array, value, getindex=False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    # return np.asarray(array)[(np.abs(array - value)).argmin()]
    if getindex:
    	return idx
    else:
    	return array[idx]

def scinote(n,digits=2):
	if n>0:
		return str(np.round(10**(np.log10(n) % 1), digits))+'e'+str(np.int(np.log10(n)))
	elif n==np.nan:
		return 'nan'
	elif n==np.inf:
		return 'inf'
	elif n==-np.inf:
		return '-inf'
	elif n==0:
		return '0'
	else:
		n = np.abs(n)
		return '-'+str(np.round(10**(np.log10(n) % 1), digits))+'e'+str(np.int(np.log10(n)))

def time(scalefactor,sim):
	h, omega_m, omega_l = cat.read(sim,0,'cosmology:hubble','cosmology:omega_matter','cosmology:omega_lambda')
	H0 = 100*h; H0G = H0*to_Gyr
	return 2. / (3.*H0G*(np.sqrt(omega_l)))*np.arcsinh( np.sqrt(omega_l/omega_m)*(scalefactor**1.5) )

def scalefactor(time, sim):
	h, omega_m, omega_l = cat.read(sim,0,'cosmology:hubble','cosmology:omega_matter','cosmology:omega_lambda')
	H0 = 100*h; H0G = H0*to_Gyr
	return ( np.sqrt(omega_m/omega_l)*np.sinh((3./2.)*H0G*np.sqrt(omega_l)*time)  )**(1./1.5)

def NFW_mass(m200, r200, c, r):
	rho_0 = (m200/(4.*np.pi*(r200/c)**3.))*(np.log(1.+c) - c/(1+c))
	m_of_r = (4*np.pi*rho_0*(r200/c)**3.) * (np.log(1+(c/r200)*r) - (c/r200)*r/(1 + (c/r200)*r))
	return m_of_r

def dicintio_parameters(Mstar,Mhalo):
	X = np.log10(Mstar/Mhalo)
	alpha = 2.94 - np.log10(	( 10**(X+2.33) )**-1.08 	+ 	( 10**(X+2.33) )**2.29		)
	beta = 4.23 + 1.34*X + 0.26*X*X
	gamma = -0.06 + np.log10(	( 10**(X+2.56) )**-0.68		+	  10**(X+2.56)				)
	return alpha,beta,gamma

def dicintio_profile(rho_s,r_s,drange,Mstar,Mhalo):
	alpha,beta,gamma = dicintio_parameters(Mstar,Mhalo)
	return rho_s / ( (drange/r_s)**gamma  *  (1 + (drange/r_s)**alpha)**((beta-gamma)/alpha) )
	
def dynamical_time(r,vc):
	kpc_to_km = 3.086e16
	tdyn = (2*np.pi*r*kpc_to_km/vc)/(3.154e7)

	return tdyn

