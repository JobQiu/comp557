import numpy as np
import matplotlib.pyplot as plt
import sys


# defining initial sequence
# aminoacid sequence
# example from our HW
a = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0]


# defining some basic functions we will need

def getCoordinate(conf, node):
	(x,y)=(0,0)
	for i in range(node):
		x += np.real(conf[i])
		y += np.imag(conf[i])
	return (x,y)

# this one is for debugging only in the current version
# in this function we don't check for intersections between nodes i and i + 1 for obvious reasons
def checkIntersections(conf):
	l = len(conf)
	for i in range(l):
		for j in range(i + 2, l + 1):
			if(getCoordinate(conf, i) == getCoordinate(conf, j)):
				return False
	return True

# in this function we actually use squared dictance as a measure of energy, which
# gives us the same behaviour and saves some computational power
def getEnergy (conf, chain):
	l = len(conf)+1
	e = 0
	for i in range(l):
		for j in range(i + 1, l):
			if(chain[i] == 1 and chain[j] == 1):
				e += (getCoordinate(conf, i)[0] - getCoordinate(conf, j)[0])**2 + (getCoordinate(conf, i)[1] - getCoordinate(conf, j)[1])**2
	return e

# makes bend after node specified
# angle code is as follows:
# 1 = 90 degrees CW
# -1 = 90 degrees CCW
def makeBend(conf, node, angle):
	for i in range(node, len(conf)):
		if(np.real(conf[i]) != 0):
			conf.imag[i] = - angle * conf.real[i]
			conf.real[i] = 0.
		else:
			conf.real[i] = angle * conf.imag[i]
			conf.imag[i] = 0.
	return conf

################################################################################
# core of the program
# the simplest zig-zag search

# this function generates U-turns in random places
# number of turns and their positions are random
# only direstion is certain - CCW first, then CW, then CCW again and so on

def fold(conf):
	bends = []
	f = True
	b = 1
	n = -1
	n = np.random.randint(n + 2, len(a) - 2)
	while(f):
		bends.append(n)
		conf = makeBend(conf, n, b)
		conf = makeBend(conf, n + 1, b)
		b *= -1
		n = np.random.randint(n + 2, len(a))
		if(n > len(a) - 3):
			f = False
	return conf, bends


# the trickiest part - shift U-turns in our folded protein and try to optimize energy
# this fill find random U-turn, will shift it by 1 in random direction (left/fight)
# and will check if new protein is better, if yes - it will accept it,
# if no - will neglect it
# this procedure is repeated N times

def refine(conf, bends):
	go = True
	max_fail = 20 # max number of consequent failed tries to improve energy
	failed = 0
	good = False # good shift or not. This function should not produce unphysical shifts.
	while(go):
		old_e = getEnergy(conf, a)
		old_conf = np.array(conf)
		while(not good):
			# generate random positions until valid is found
			b = np.random.randint(len(bends)) # num of bend
			n = bends[b] # num of node
			d = np.random.choice([-1, 1]) # direction
			if(b == 0 and n == 0 and d == -1):
				good = False
				continue
			elif(b == (len(bends) - 1) and n == (len(conf) - 1) and d == 1):
				good = False
				continue
			elif(n + d > len(conf) or n - d > len(conf)):
				good = False
				continue
			elif(b + d > (len(bends) - 1) or (b - d > len(bends) - 1)):
				good = False
				continue
			elif(abs(bends[b+d] - bends[b]) <= 2):
				good = False
				continue
			else:
				conf[n + d] = conf[n]
				conf[n] = conf[n - d]
				bends[b] += d
				good = True
		if(getEnergy(conf, a) > old_e):
			conf = old_conf
			bends[b] -= d
			good = False
			failed += 1
		else:
			good = False
			failed = 0
		if(failed == max_fail):
			go = False
	return conf

best_conf = np.ones(33,dtype=np.complex_);
best_ener = getEnergy(best_conf, a)
sys.stdout.flush()




# main cycle

for i in range(25):
	print "Iteration ",i+1
	sys.stdout.flush()
	bends = []
	while(len(bends) < 5):
		c, bends = fold(np.ones(33,dtype=np.complex_))
	c = refine(c, bends)
	if(getEnergy(c, a) < best_ener):
		best_conf = np.array(c)
		best_ener = getEnergy(best_conf, a)
		c = best_conf
		# plot the intermediate result
		plt.clf()
		for i in range(len(a)):
			if(i != (len(a) - 1)):
				plt.plot([getCoordinate(c, i)[0], getCoordinate(c, i + 1)[0]], [getCoordinate(c, i)[1], getCoordinate(c, i + 1)[1]], 'k-')
			if(a[i] == 0):
				plt.plot(getCoordinate(c, i)[0],getCoordinate(c, i)[1], 'ro')
			else:
				plt.plot(getCoordinate(c, i)[0],getCoordinate(c, i)[1], 'go')
		plt.axis([-10, 10, -10, 10])
		#plt.axis('off')
		plt.savefig('curr_best.png')
		print "New best score: ",best_ener
		sys.stdout.flush()