#import libraries
import numpy as np 
import matplotlib.pyplot as plt 
import plot_helper as p
import pdb

#define your plotting function
def example():
	#create figure and set border width
	fig,ax = p.makefig(1)
	plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':'0.05'})

	#create example data
	x_data = np.arange(100)
	y1_data = np.arange(100)*(1 + np.random.random(100))
	y2_data = np.arange(100)*(1 + 1.5*np.random.random(100))

	#plot the data
	ax.plot(x_data,y1_data,ls='-',marker='o',ms=2,lw=1.8,c='teal',alpha=0.7,label='data set 1')
	ax.plot(x_data,y2_data,ls='-',marker='o',ms=2,lw=1.8,c='orange',alpha=0.7,label='data set 2')

	# ls = line style
	# ms = marker size
	# lw = line width
	# c = color
	# alpha = transparency

	#turn on the legend
	ax.legend(frameon=False,prop={'size':11})
	# turned off frame, set font size = 11

	ax.set_xlabel('x data [unit]')
	ax.set_ylabel('y data [unit]')
	# note that you can use latex formatting with an 'r' before the string. example:
		# ax.set_xlabel(r'x data [$M_\odot$]')

	'''
	other useful functions include
	ax.set_xticks([1,2,3]) 				-> tells matplotlib where to place ticks
										-> you can also include minor=True after the array to set minor tick locations
	ax.set_ticklabels(['1','2','3']) 	-> tells matplotlib what to call those ticks
	ax.set_xlim(0,100) 					-> sets the boundaries of the axis
	ax.set_xscale('log')				-> changes axis scaling to log_10

	'''

	plt.show()
	# plt.savefig('/mydirectory/example_plot.png',format=png,dpi=200)
	# you can change the dpi (dots per inch) to change how HD the figure is. 200 is a good amount
	# you can also change the format to other things like pdf (for example)

#call your plotting function
example()

#call the debugger
pdb.set_trace()