import snapHDF5 as snap
import numpy as np
import matplotlib.pyplot as plt
import pdb 

dir = "/media/ethan/Watson/research/data/metaldiff/res880/"

# num = 500 # 132 
# snap_name = "snapshot_"

# fname = Base_dir + sim_name + "/" + snap_name + "%03d" % (num)

fname = dir+"snapshot_600"

## Example: get info from header:
# header = snap.snapshot_header(fname)
# Ngas = header.nall[0]
# print("Num of gas=%s" % Ngas)
# ##----

# ## Example, read gas density:
# rho = snap.read_block(fname,"RHO ",parttype=0)

## Example, read stellar positions and mass:
pos_str = snap.read_block(fname,"POS ",parttype=4)   #NOTE: get the 4-char flag "POS ", from the snappHDF5.py list
# mass_str = snap.read_block(fname,"MASS",parttype=4)

plt.plot(pos_str[:,0],pos_str[:,2],',', alpha=0.2)
plt.title("x-z")
plt.show()

pdb.set_trace()
