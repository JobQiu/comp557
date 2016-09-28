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
    mdp.computeStates()
    if len(V) == 0:
	V = {k: 0 for k in mdp.states}
    vT_1 = V.copy()
    vT = {}
    for s in V.keys():
	value = 0
	for tup in mdp.succAndProbReward(s, pi[s]):
	    value += tup[1] * (tup[2] + (mdp.discount() * vT_1[tup[0]]))
	vT[s] = value
    while max(abs(x - y) for x,y in zip(vT.values(), vT_1.values())) > epsilon:
	vT_1 = vT.copy()
	vT = {}
	for s in V.keys():
	    value = 0
	    for tup in mdp.succAndProbReward(s, pi[s]):
		value += tup[1] * (tup[2] + (mdp.discount() * vT_1[tup[0]]))
	    vT[s] = value

    return vT
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
	V = {k:0 for k in mdp.states}
	pi = {k:random.choice(mdp.actions(k)) for k in mdp.states}
	unConverged = True
	while unConverged:
	    vT_1 = V.copy()
	    V = {k:0 for k in mdp.states}
	    for s in mdp.states:
		value = 0
		for tup in mdp.succAndProbReward(s, pi[s]):
		    value += tup[1] * (tup[2] + (mdp.discount() * vT_1[tup[0]]))
		V[s] = value
	    pi = computeOptimalPolicy(mdp, V)
	    if max(abs(x - y) for x,y in zip(V.values(), vT_1.values())) <= epsilon:
		unConverged = False
		
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
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    raise Exception("Not implemented yet")
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
        raise Exception("Not implemented yet")
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3.2.2

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

