import numpy as np
import matplotlib.pyplot as plt
import pdb
import pylab as pl

plt.clf()

cm = plt.cm.get_cmap('RdYlBu')


# xy = range(20)
# z = xy
# sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=20, s=35, cmap=cm)

x = np.array(range(20))
y = np.array(range(20))
z = np.random.random(20)*10
sc = plt.scatter(x, z, c=y, vmin=0, vmax=20, s=9, cmap=cm, alpha=0.5, edgecolors='none')
plt.grid(True, alpha=0.1)

plt.colorbar(sc)
#plt.show()

pl.savefig('/home/ejahn003/plots/color.png',format='png',dpi=500)

pdb.set_trace()
