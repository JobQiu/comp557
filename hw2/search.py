import re, searchutil


############################################################
# Problem 1b: A-star search

# Takes the SearchProblem |problem| you're trying to solve and a |heuristic|
# (which is a function that maps a state to an estimate of the cost to the
# goal).  Returns another search problem |newProblem| such that running uniform
# cost search on |newProblem| is equivalent to running A* on |problem| with
# |heuristic|.
def astarReduction(problem, heuristic):
    class NewSearchProblem(searchutil.SearchProblem):
        # Please refer to searchutil.SearchProblem to see the functions you need to
        # overried.
        # BEGIN_YOUR_CODE (around 9 lines of code expected)
        def __init__(self): self.start = problem.startState()
        
        def startState(self):
            return self.start
        
        def isGoal(self, state):
            return problem.isGoal(state)
            
        def succAndCost(self, state):
            edges = []
            for (action, newState, cost) in problem.succAndCost(state):
                edges.append((action, newState, cost + heuristic(newState) - heuristic(state)))
            return edges
        # END_YOUR_CODE
    newProblem = NewSearchProblem()
    return newProblem

# Implements A-star search by doing a reduction.
class AStarSearch(searchutil.SearchAlgorithm):
    def __init__(self, heuristic):
        self.heuristic = heuristic

    def solve(self, problem):
        # Reduce the |problem| to |newProblem|, which is solved by UCS.
        newProblem = astarReduction(problem, self.heuristic)
        algorithm = searchutil.UniformCostSearch()
        algorithm.solve(newProblem)

        # Copy solution back
        self.actions = algorithm.actions
        if algorithm.totalCost != None:
            self.totalCost = algorithm.totalCost + self.heuristic(problem.startState())
        else:
            self.totalCost = None
        self.numStatesExplored = algorithm.numStatesExplored

############################################################
# Problem 2b: Delivery

class DeliveryProblem(searchutil.SearchProblem):
    # |scenario|: delivery specification.
    def __init__(self, scenario):
        self.scenario = scenario
        self.start = (self.scenario.truckLocation, tuple(list([0] * self.scenario.numPackages)))
    # Return the start state.
    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return self.start
        # END_YOUR_CODE

    # Return whether |state| is a goal state or not.
    def isGoal(self, state):
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        return list(state[1]) == [1] * self.scenario.numPackages
        # END_YOUR_CODE

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state):
        # Hint: Call self.scenario.getNeighbors((x,y)) to get the valid neighbors
        # at that location. In order for the simulation code to work, please use
        # the exact strings 'Pickup' and 'Dropoff' for those two actions.
        # BEGIN_YOUR_CODE (around 18 lines of code expected)
        edges = []
        # find out type of our location
        if(state[0] in self.scenario.pickupLocations):
            # we are on the pickup location
            for i in range(self.scenario.numPackages):
                if self.scenario.pickupLocations[i] == state[0]:
                    n = i
            if(list(state[1])[n] == 0):
                new_stat = list(state[1])
                new_stat[n] = 1
                edges.append(('Pickup', (state[0], tuple(new_stat)), 0))
        for i in self.scenario.getNeighbors(state[0]):
            edges.append((i[0], (i[1], state[1]), 1))
        return edges
        # END_YOUR_CODE

# Return a heuristic corresponding to solving a relaxed problem
# where you can ignore all barriers, but
# you'll need to deliver the given |package|, and then go home
def createHeuristic2(scenario):
    def heuristic(state):
        # BEGIN_YOUR_CODE (around 11 lines of code expected)
        heuristic = 0
        me = state[0]
        pickup = scenario.pickupLocations[0]
        heuristic += (abs(me[0] - pickup[0]) + abs(me[1] - pickup[1]))
        return heuristic
        # END_YOUR_CODE
    return heuristic

