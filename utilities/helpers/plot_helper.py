from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime
import directories as d
# import math_helper as m
import mycolors as c
import os,sys,inspect


hostname = os.uname()[1]
monthnum = int(str(datetime.now()).split(' ')[0].split('-')[1]) - 1
monthlist = ['01.jan','02.feb','03.mar','04.apr','05.may','06.jun','07.jul','08.aug','09.sep','10.oct','11.nov','12.dec']
month = monthlist[monthnum]

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
s=16 				# regular font size,
ls=18				# label font size
lw=2.5  			# line width
ms=7				# marker size
gs=11				# legend font size
aph=0.7 			# alpha
ntl=3				# minor tick length
jtl=6				# major tick length
ts=11				# overlay text size
ra=-90				# text rotation angle
ms=4.5				# marker size
grey = '#727d8e' 	# the color grey

#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
class MplColorHelper:
	def __init__(self, cmap_name, start_val, stop_val):
		self.cmap_name = cmap_name
		self.cmap = plt.get_cmap(cmap_name)
		self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
		self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
	def get_rgb(self, val):
		return self.scalarMap.to_rgba(val)

def makefig(n_panels=1,height=2.5,figx=6,figy=6):
	if n_panels==1:
		fig, ax = plt.subplots(figsize=(figx,figy))
		plt.rc('font', family='serif', size=s)
		ax.tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
				left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,
				length=ntl)#,zorder=100)
		ax.tick_params(axis='both',which='major',length=jtl)
		return fig, ax

	elif n_panels==2:
		fig, axarr = plt.subplots(2, sharex=True, gridspec_kw = {'height_ratios':[height, 1]},figsize=(figx,figy)) 
		# plt.rc('text', usetex=True)
		plt.rc('font', family='serif',size=s)

		axarr[0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)
		axarr[1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)
		axarr[0].tick_params(axis='both',which='major',length=jtl)
		axarr[1].tick_params(axis='both',which='major',length=jtl)
		return fig, axarr
	
	elif n_panels=='3_vert':
		plt.rc('font', family='serif',size=s)
		fig = plt.figure(figsize=(10,7))
		ax = fig.add_subplot(111)    # The big subplot
		ax1 = fig.add_subplot(311)
		ax2 = fig.add_subplot(312)
		ax3 = fig.add_subplot(313)
		axarr = [ax1,ax2,ax3]

		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


		fig.subplots_adjust(wspace=0)
		fig.subplots_adjust(hspace=0)
		return fig, ax, axarr
	elif n_panels=='3_horiz':
		fig, axarr = plt.subplots(nrows=1, ncols=3, sharey='row', figsize=(figx,figy)) 
		# plt.rc('text', usetex=True)
		fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)
		plt.rc('font', family='serif',size=s)

		axarr[0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=False, right=False, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[2].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=False, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[0].tick_params(axis='both',which='major',length=jtl)
		axarr[1].tick_params(axis='both',which='major',length=jtl)
		axarr[2].tick_params(axis='both',which='major',length=jtl)

		return fig, axarr



	elif n_panels==6:
		fig, axarr = plt.subplots(nrows=2,ncols=3,sharex='col',sharey='row',
			gridspec_kw = {'height_ratios':[height, 1], 'width_ratios':[1,1,1]},figsize=(15,6) )
		fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)
		plt.rc('font', family='serif', size=s)

		#--------
		axarr[0,0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[0,1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[0,2].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		#--------
		axarr[1,0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[1,1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[1,2].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		for i in range(0,2):
			for j in range(0,3):
				axarr[i,j].tick_params(axis='both',which='major',length=jtl)
				# axarr[i,j].text(0.5,0.5,str(i)+','+str(j))
		return fig, axarr

	elif n_panels==8:
		fig, axarr = plt.subplots(nrows=2,ncols=4,sharex='col',sharey='row',
			gridspec_kw = {'height_ratios':[height, 1], 'width_ratios':[1,1,1,1]},figsize=(16,8) )
		fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)
		plt.rc('font', family='serif', size=s)

		#--------
		axarr[0,0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[0,1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[0,2].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[0,3].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		#--------
		axarr[1,0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[1,1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[1,2].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[1,3].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		#--------
		for i in range(0,2):
			for j in range(0,4):
				axarr[i,j].tick_params(axis='both',which='major',length=jtl)
				# axarr[i,j].text(0.5,0.5,str(i)+','+str(j))

		# plt.rc('font', family='serif',size=s)
		# fig = plt.figure(figsize=(10,7))
		# ax = fig.add_subplot(111)    # The big subplot
		# ax1 = fig.add_subplot(421)
		# ax2 = fig.add_subplot(422)
		# ax3 = fig.add_subplot(423)
		# ax4 = fig.add_subplot(424)
		# ax5 = fig.add_subplot(425)
		# ax6 = fig.add_subplot(426)
		# ax7 = fig.add_subplot(427)
		# ax8 = fig.add_subplot(428)
		# axarr = [[ax1,ax2,ax3,ax4],[ax5,ax6,ax7,ax8]]

		# ax.spines['top'].set_color('none')
		# ax.spines['bottom'].set_color('none')
		# ax.spines['left'].set_color('none')
		# ax.spines['right'].set_color('none')
		# ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

		# fig.subplots_adjust(wspace=0)
		# fig.subplots_adjust(hspace=0)
		# return fig, ax, axarr


		#--------
		return fig, axarr

	elif n_panels==11:
		fig, axarr = plt.subplots(nrows=3,ncols=4,sharex=True,sharey=True,figsize=(15,12))
		plt.rc('font', family='serif', size=s)
			#gridspec_kw = {'height_ratios':[height, 1], 'width_ratios':[1,1,1]},
		fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)
		
		#--------
		for i in range(0,3):
			for j in range(0,4):
				axarr[i,j].tick_params(axis='both',which='major',length=jtl)
				# axarr[i,j].text(0.5,0.5,str(i)+','+str(j))
		fig.delaxes(axarr[2,3])

		#--------
		axarr[0,0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[0,1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[0,2].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[0,3].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		#--------
		axarr[1,0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[1,1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[1,2].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=False, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[1,3].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		#--------
		axarr[2,0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[2,1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		axarr[2,2].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)


		return fig, axarr

	elif n_panels=='3_proj':
		fig, axarr = plt.subplots(nrows=2,ncols=2,sharex='col',sharey='row',figsize=(10,10))
		fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)
		plt.rc('font', family='serif', size=s)

		#--------
		for i in range(0,2):
			for j in range(0,2):
				axarr[i,j].tick_params(axis='both',which='major',length=jtl)
				# axarr[i,j].text(0.5,0.5,str(i)+','+str(j))
		fig.delaxes(axarr[0,1])

		return fig, axarr

	elif n_panels=='2_proj':
		fig, axarr = plt.subplots(nrows=1,ncols=2,figsize=(10,5))
		fig.subplots_adjust(wspace=0);fig.subplots_adjust(hspace=0)
		plt.rc('font', family='serif', size=s)

		axarr[0].tick_params(axis='both', which='both', top=False, bottom=False, labelbottom=False, labeltop=False,
			left=False, right=False, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)
		axarr[1].tick_params(axis='both', which='both', top=False, bottom=False, labelbottom=False, labeltop=False,
			left=False, right=False, labelleft=False, labelright=False, direction='in',labelsize=s,length=ntl)

		return fig, axarr

	elif n_panels=='density thing':
		fig, axarr = plt.subplots(nrows=2,ncols=1,figsize=(7,10))
		plt.rc('font', family='serif', size=s)

		axarr[0].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)
		axarr[1].tick_params(axis='both', which='both', top=True, bottom=True, labelbottom=True, labeltop=False,
			left=True, right=True, labelleft=True, labelright=False, direction='in',labelsize=s,length=ntl)

		return fig, axarr

	else:
		raise ValueError('unknown panel configuration')
	
def finalize(fig,fname='',save=False,tight=True,pad='0.03',save_pdf=False,dpi=200):
	if fname=='':
		# print(inspect.stack())
		# for i in np.arange(len(inspect.stack())):
		# 	print(inspect.stack()[i],'\n')

		# print(inspect.stack()[1][3],'\n')

		if hostname=='master':
			fname = inspect.stack()[1][3]
		elif hostname=='peregrin' or hostname=='eostrix':
			fname = inspect.stack()[1].function
		else:
			raise ValueError('I don\'t know how to read "inspect.stack()" on this host')
		

	if save:
		print('Saving figure: '+fname)
		plt.rcParams.update({'savefig.bbox': 'tight', 'savefig.pad_inches':pad})
		plt.savefig(d.plotdir+month+'/'+fname.replace(' ','_')+'.png',dpi=dpi,format='png')
		if save_pdf:
			plt.savefig(d.plotdir+month+'/pdfs/'+fname.replace(' ','_')+'.pdf',format='pdf')
		plt.close(fig)

	else:
		print('Showing figure: '+fname)
		if tight:
			plt.tight_layout()
		plt.show()

def clear_axes(ax):
	from matplotlib.ticker import StrMethodFormatter, NullFormatter
	ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	ax.yaxis.set_minor_formatter(NullFormatter())
	ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
	ax.xaxis.set_minor_formatter(NullFormatter())




#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------