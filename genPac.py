#!/usr/bin/env python

# Searches for optimal Pac genes using a genetic algorithm.
# Inspiration from http://www-cs-students.stanford.edu/~jl/Essays/ga.html

import _PyPacwar
import numpy


# parameters
popSize = 50
numElites = 10
mutationRate = 0.002
numGenerations = 1000
randSeed = 2501

# global variables
currentPop, bestGene, prevBestGene = [], [], []
scores = ranks = [0]*popSize


def initializePopulation(n):

	global bestGene
	for i in range(n):
		numpy.random.seed(randSeed * i)
		currentPop.append(numpy.random.randint(4, size=50).tolist())

	bestGene = list(currentPop[numpy.random.randint(50)])



def evaluatePopulation():

	global ranks, bestGene, prevBestGene
#	print "Best:", bestGene
	for g1 in range(popSize):
		for g2 in range(g1, popSize):
			if (g1 == g2): continue
			(s1, s2) = scoreDuel(_PyPacwar.battle(currentPop[g1], currentPop[g2]))
			scores[g1] += s1
			scores[g2] += s2
#	print scores
#	print sum(scores)
	#scores.sort(reverse = True) # sorts scores in place
#	print sorted(scores, reverse = True)
	ranks = sorted(range(len(scores)), reverse = True, key = lambda x: scores[x])
#	print ranks
#	print [scores[i] for i in ranks[:numElites]]
	prevBestGene = list(bestGene)
	bestGene = list(currentPop[ranks[0]])
#	print "New best:", bestGene
#	print currentPop[ranks[0]]



def createNextGen():

	global currentPop, scores
	nextPop = [currentPop[i] for i in ranks[:numElites]] # carry over top genes
	
	for i in range(popSize - numElites):
		newGene = [0]*50
		parentOne = selectGene()
		parentTwo = selectGene()
		crossPoint = numpy.random.randint(50)
		
		for bp in range(50):
			if (bp < crossPoint):
				newGene[bp] = currentPop[parentOne][bp]
			else:
				newGene[bp] = currentPop[parentTwo][bp]
			if (numpy.random.random() < mutationRate):
				others = [x for x in range(4) if x != newGene[bp]]
				newGene[bp] = numpy.random.choice(others)
				
		nextPop.append(newGene)

	currentPop = list(nextPop)
	scores = [0]*popSize


# use roulette wheel selection
def selectGene():

	run = 0
	randPoint = numpy.random.randint(sum(scores))
	for i in range(popSize):
		run += scores[i]
		if (run >= randPoint):
			return i



def scoreDuel((rounds, c1, c2)):

	s1 = s2 = margin = 0
	if (c1 != 0 and c2 != 0): margin = max(c1, c2) / min(c1, c2)
	if (c1 == 0):
		if (rounds < 100):
			s2 = 20
		elif (rounds < 200):
			s1 = 1
			s2 = 19
		elif (rounds < 300):
			s1 = 2
			s2 = 18
		else:
			s1 = 3
			s2 = 17
	elif (c2 == 0):
		if (rounds < 100):
			s1 = 20
		elif (rounds < 200):
			s1 = 19
			s2 = 1
		elif (rounds < 300):
			s1 = 18
			s2 = 2
		else:
			s1 = 17
			s2 = 3
	elif (c1 > c2):
		if (margin >= 10):
			s1 = 13
			s2 = 7
		elif (margin > 3):
			s1 = 12
			s2 = 8
		elif (margin > 1.5):
			s1 = 11
			s2 = 9
		else:
			s1 = s2 = 10
	else:
		if (margin >= 10):
			s1 = 7
			s2 = 13
		elif (margin > 3):
			s1 = 8
			s2 = 12
		elif (margin > 1.5):
			s1 = 9
			s2 = 11
		else:
			s1 = s2 = 10

	return (s1, s2)

	
def main():

	ones   = [1]*50
	threes = [3]*50

	initializePopulation(popSize)
	evaluatePopulation()
	(rounds,c1,c2) = _PyPacwar.battle(prevBestGene, bestGene)
	print "Number of rounds:", rounds 
	print "pB PAC-mites remaining:", c1
	print "B PAC-mites remaining:", c2

	(rounds,c1,c2) = _PyPacwar.battle(ones, bestGene)
	print "Number of rounds:", rounds 
	print "Ones PAC-mites remaining:", c1
	print "B PAC-mites remaining:", c2

	(rounds,c1,c2) = _PyPacwar.battle(threes, bestGene)
	print "Number of rounds:", rounds 
	print "Threes PAC-mites remaining:", c1
	print "B PAC-mites remaining:", c2

	for i in range(numGenerations):
		createNextGen()
		evaluatePopulation()
		(rounds,c1,c2) = _PyPacwar.battle(prevBestGene, bestGene)
		print "Number of rounds:", rounds 
		print "pB PAC-mites remaining:", c1
		print "B PAC-mites remaining:", c2

	(rounds,c1,c2) = _PyPacwar.battle(ones, bestGene)
	print "Number of rounds:", rounds 
	print "Ones PAC-mites remaining:", c1
	print "B PAC-mites remaining:", c2

	(rounds,c1,c2) = _PyPacwar.battle(threes, bestGene)
	print "Number of rounds:", rounds 
	print "Threes PAC-mites remaining:", c1
	print "B PAC-mites remaining:", c2

	print "Best gene:", bestGene
	"""
	print "Example Python module in C for Pacwar"
	print "all ones versus all threes ..."
	(rounds,c1,c2) = _PyPacwar.battle(ones, threes)
	print "Number of rounds:", rounds 
	print "Ones PAC-mites remaining:", c1
	print "Threes PAC-mites remaining:", c2
	"""

if __name__ == "__main__": main()
