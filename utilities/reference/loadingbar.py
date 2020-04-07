import time
import sys
import pdb
import os
import numpy as np
import random
rows, columns = os.popen('stty size', 'r').read().split()
rows = np.int(rows)
columns = np.int(columns)

print('\nsome stuff')

# setup progress bar
sys.stdout.write("[%s]" % (" " * (columns-2)))
sys.stdout.flush()
sys.stdout.write("\b" * (columns-2)) 
printThing='-'
n_prev = 0


loopSize = random.randrange(100,500)
array = np.random.random(1000)

doLoopSize=False

if doLoopSize:
	for i in range(loopSize):
		time.sleep(0.005) # do real work here
		i = np.float(i)
		n = np.int((columns-1)*(i/np.float(loopSize)))/len(printThing)

		if not(n==n_prev):
			sys.stdout.write(printThing)
			sys.stdout.flush()
			n_prev = n
		else:
			continue
else:
	for i in range(len(array)):
		time.sleep(0.005) # do real work here

		
		# i = np.float(i)
		n = np.int((columns-1)*(i/np.float(len(array))))/len(printThing)

		if not(n==n_prev):
			sys.stdout.write(printThing)
			sys.stdout.flush()
			n_prev = n
		else:
			continue



sys.stdout.write("\n")

print('some other stuff\n')


pdb.set_trace()