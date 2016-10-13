import collections, util, math, random

############################################################
# Problem 3.1.1

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).  Note that |V| is a dictionary.  
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    que = 0
    for tup in mdp.succAndProbReward(state, action):
	que += tup[1] * (tup[2] + (mdp.discount() * V[tup[0]]))
    return que
    # END_YOUR_CODE


############################################################
# Problem 3.1.2

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.  Note that |V| and |pi| are
    dictionaries.
    """
    # BEGIN_YOUR_CODE (around 8 lines of code expected)
    lastV = collections.defaultdict(int)

    for i in range(100000):  # ok?
	for s in mdp.states:
	    V[s] = 0
	    a = pi[s]
	    for nS, prob, reward in mdp.succAndProbReward(s, a):
		V[s] += prob * (reward + mdp.discount() * lastV[nS])

	a = maxDiff(V, lastV)
	if a < epsilon:
	    break

	lastV = V.copy()

    return V

    # END_YOUR_CODE

############################################################
# Problem 3.1.3

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().  Note that |V| is a
    dictionary.
    """
    # BEGIN_YOUR_CODE (around 4 lines of code expected)
    pi = {}
    for s in V.keys():
	possibles = {}
	for a in mdp.actions(s):
	    possibles[a] = computeQ(mdp, V, s, a)
	pi[s] = max(possibles)
    return pi
    # END_YOUR_CODE

############################################################
# Problem 3.1.4

class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # compute |V| and |pi|, which should both be dicts
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
	V = {k:0 for k in mdp.states}
	pi = {k:random.choice(mdp.actions(k)) for k in mdp.states}
	newPolicy = True
	while newPolicy:
	    V = policyEvaluation(mdp, V, pi, epsilon)
	    newPolicy = False
	    pip = computeOptimalPolicy(mdp, V)
	    if cmp(pi, pip):
		pi = pip.copy()
		newPolicy = True
		    
        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 3.1.5

class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        # BEGIN_YOUR_CODE (around 11lines of code expected)
	V = collections.defaultdict(int)
	lastV = V.copy()
	V1 = V.copy()

	for i in range(100000):  # ok?
	    for s in mdp.states:
		V1[s] = max(computeQ(mdp, lastV, s, a) for a in mdp.actions(s))

	    if maxDiff(lastV, V1) < epsilon:
		break

	    lastV = V1.copy()
	V = V1

	pi = computeOptimalPolicy(mdp, V)

        # END_YOUR_CODE
        self.pi = pi
        self.V = V

############################################################
# Problem 3.1.6

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(util.MDP):
    def __init__(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return none
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return none
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
	if state == 1:
	    return []
	if state == 0:
	    return ['Stay', 'Quit']
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        if state == 1:
	    return []
	if state == 0:
	    if action == 'Stay':
		return [(0, 1/3, 4), (1, 2/3, 4)]
	    if action == 'Quit':
		return [(1, 1, 5)]
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 1
        # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 1/3
    # END_YOUR_CODE

############################################################
# Problem 3.2.1

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.  The second element is the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.  The final element
    # is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to (0,).
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 40 lines of code expected)
	hand, nextCard, deckMult = state
	possibles = []
	if deckMult == (0,):
	    return []
	
	### player decides to 'Take'
	if action == 'Take':
	    if nextCard != None:	# peeked last time
		hand += self.cardValues[nextCard]
		prob = 1
		reward = 0
		if hand > self.threshold:	# bust
		    newState = (hand, None, (0,))
		else:
		    newDeckMult = list(deckMult)
		    newDeckMult[nextCard] -= 1
		    if sum(newDeckMult) <= 0:
			reward = hand
			newState = (hand, None, (0,))
		    else:
			newState = (hand, None, tuple(newDeckMult))
		possibles.append((newState, prob, reward))
	    else:			# didn't peek last time
		sumPossibles = sum(deckMult)
		for i in range(len(self.cardValues)):
		    newDeckMult = list(deckMult)
		    prob = newDeckMult[i] / float(sumPossibles)
		    if prob == 0: continue
		    reward = 0

		    newHand = hand + self.cardValues[i]
		    if newHand > self.threshold:	# bust
			newState = (newHand, None, (0,))
		    else:
			assert (newDeckMult[i] > 0)
			newDeckMult[i] -= 1
			if sum(newDeckMult) <= 0:
			    reward = newHand
			    newState = (newHand, None, (0,))
			else:
			    newState = (newHand, None, tuple(newDeckMult))
		    possibles.append((newState, prob, reward))

	### player decides to 'Peek'
	if action == 'Peek':
	    if nextCard != None:  # can't peek twice in a row
		return []
	    else:
		sumPossibles = sum(deckMult)
		for i in range(len(self.cardValues)):
		    newDeckMult = list(deckMult)
		    prob = newDeckMult[i] / float(sumPossibles)
		    if prob == 0: continue

		    reward = -self.peekCost
		    newState = (hand, i, tuple(newDeckMult))
		    possibles.append((newState, prob, reward))

	### player decides to 'Quit'
	if action == 'Quit':
	    newState = (hand, None, (0,))
	    prob = 1
	    reward = hand
	    possibles.append((newState, prob, reward))

	return possibles
        # END_YOUR_CODE

    def discount(self):
        return 1

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            #if state[2] == (0,):  continue
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
		    #if reward == (0,): continue
		    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)

############################################################
# Problem 3.2.2

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    return BlackjackMDP([1,2,3,4,5,6,20], 3, 20, 1)
    # END_YOUR_CODE


###################
### helper function
def maxDiff(v1, v2):
    tmp = []
    for k in v1:
	tmp.append(abs(v1[k] - v2[k]))
    return max(tmp)
###################




