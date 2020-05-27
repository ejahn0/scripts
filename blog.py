import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

%matplotlib inline
rnd = np.random.RandomState(seed=42)

n_data = 16 # number of data points
a_true = 1.255 # randomly chosen truth
b_true = 4.507 

x = rnd.uniform(0, 2., n_data)