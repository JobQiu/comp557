from util import manhattanDistance
from game import Directions
import random, util, sys, searchutil, search

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)
    self.last = None # for smarter rule of breaking ties

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
    
    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean that your algorithm is correct.
  """

  
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
	
      gameState.isWin():
        Returns True if it's a winning state
	
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
        
      It is recommended you have separate functions: value(), max_value(), and min_value() as in the slides
      and call these functions here to make your code understandable.
    """
    
    # BEGIN_YOUR_CODE (around 35 lines of code expected)
    def value(state, depth, index):
        if(state.isWin() or state.isLose() or depth == 0):
            return self.evaluationFunction(state)
        elif(index == 0):
            return max_value(state, depth)
        elif(index != 0):
            return min_value(state, depth, index)
    
    def max_value(state, depth):
        v = float("-inf")
        for i in state.getLegalActions(0):
            v = max(v, value(state.generateSuccessor(0, i), depth, 1))
        return v
    
    def min_value(state, depth, index):
        v = float("inf")
        for i in state.getLegalActions(index):
            if(index == (state.getNumAgents() - 1)):
                v = min(v, value(state.generateSuccessor(index, i), depth - 1, 0))
            else:
                v = min(v, value(state.generateSuccessor(index, i), depth, index + 1))
        return v
    
    scores = [value(gameState.generateSuccessor(self.index, i), self.depth, self.index + 1) for i in gameState.getLegalActions(self.index)]
    best = max(scores)
    bestInd = [i for i in range(len(scores)) if scores[i] == best]
    bestLen = len(bestInd)
    for i in bestInd:
        if(gameState.getLegalActions(self.index)[i] == self.last and self.last != Directions.STOP and bestLen == 3):
            # continue moving this way if it is both possible and optimal and we are in the corridor (e.g. we have exactly two possible directions)
            return self.last
    chosenIndex = random.choice(bestInd)
    self.last = gameState.getLegalActions(self.index)[chosenIndex]
    return self.last
    # END_YOUR_CODE

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
    
    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean your algorithm is correct.
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
      
      The same methods used in MinimaxAgent should also be useful here   
      
      It is recommended you have separate functions: value(), max_value(), and min_value() as in the slides
      and call these functions here to make the code clear   
    """

    # BEGIN_YOUR_CODE (around 45 lines of code expected)
    def value(state, depth, index, alf, bet):
        if(state.isWin() or state.isLose() or depth == 0):
            return self.evaluationFunction(state)
        elif(index == 0):
            return max_value(state, depth, alf, bet)
        elif(index != 0):
            return min_value(state, depth, index, alf, bet)
    
    def max_value(state, depth, alf, bet):
        v = float("-inf")
        for i in state.getLegalActions(0):
            v = max(v, value(state.generateSuccessor(0, i), depth, 1, alf, bet))
            if (v >= bet):
                return v
            else:
                alf = max(alf, v)
        return v
    
    def min_value(state, depth, index, alf, bet):
        v = float("inf")
        for i in state.getLegalActions(index):
            if(index == (state.getNumAgents() - 1)):
                v = min(v, value(state.generateSuccessor(index, i), depth - 1, 0, alf, bet))
                if(v <= alf):
                    return v
                else:
                    bet = min(bet, v)
            else:
                v = min(v, value(state.generateSuccessor(index, i), depth, index + 1, alf, bet))
                if(v <= alf):
                    return v
                else:
                    bet = min(bet, v)
        return v
    
    scores = [value(gameState.generateSuccessor(self.index, i), self.depth, self.index + 1, float("-inf"), float("inf")) for i in gameState.getLegalActions(self.index)]
    best = max(scores)
    bestInd = [i for i in range(len(scores)) if scores[i] == best]
    bestLen = len(bestInd)
    for i in bestInd:
        if(gameState.getLegalActions(self.index)[i] == self.last and self.last != Directions.STOP and bestLen == 3):
            # continue moving this way if it is both possible and optimal and we are in the corridor (e.g. we have exactly two possible directions)
            return self.last
    chosenIndex = random.choice(bestInd)
    self.last = gameState.getLegalActions(self.index)[chosenIndex]
    return self.last
    # END_YOUR_CODE

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
    
    The auto grader will check the running time of your algorithm. Friendly reminder: passing the auto grader
    does not necessarily mean your algorithm is correct.
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
      
      The same methods used in MinimaxAgent should also be useful here   
      
      It is recommended you have separate functions: value(), max_value(), and expect_value() as in the slides
      and call these functions here to make the code clear
    """

    # BEGIN_YOUR_CODE (around 35 lines of code expected)
    def value(state, depth, index):
        if(state.isWin() or state.isLose() or depth == 0):
            return self.evaluationFunction(state)
        elif(index == 0):
            return max_value(state, depth)
        elif(index != 0):
            return min_value(state, depth, index)
    
    def max_value(state, depth):
        v = float("-inf")
        for i in state.getLegalActions(0):
            v = max(v, value(state.generateSuccessor(0, i), depth, 1))
        return v
    
    def min_value(state, depth, index):
        v = 0
        for i in state.getLegalActions(index):
            if(index == (state.getNumAgents() - 1)):
                v += value(state.generateSuccessor(index, i), depth - 1, 0)
            else:
                v += value(state.generateSuccessor(index, i), depth, index + 1)
        return v / len(state.getLegalActions(index))
    
    scores = [value(gameState.generateSuccessor(self.index, i), self.depth, self.index + 1) for i in gameState.getLegalActions(self.index)]
    #print gameState.getLegalActions(self.index)
    #print scores
    #sys.stdout.flush()
    best = max(scores)
    bestInd = [i for i in range(len(scores)) if scores[i] == best]
    bestLen = len(bestInd)
    for i in bestInd:
        if(gameState.getLegalActions(self.index)[i] == self.last and self.last != Directions.STOP and bestLen == 3):
            # continue moving this way if it is both possible and optimal and we are in the corridor (e.g. we have exactly two possible directions)
            return self.last
    chosenIndex = random.choice(bestInd)
    self.last = gameState.getLegalActions(self.index)[chosenIndex]
    #print self.last
    #sys.stdout.flush()
    return self.last
    # END_YOUR_CODE

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (problem 4).

    DESCRIPTION:
    
    As a base we use standard evaluation function and then we correct it a bit.
    Thus, we we are able to get the best from both approaches.
    
    In our evaluation part we have three corrections to the state score:
    
    1. First correction is the distance to the nearest "goodie" - that is everything,
    that can be eaten.
    This can be the food, capsule or even scared ghost. This decrees final score:
    the bigger the distance, the lower the score. However, this part has low coefficient
    of 0.05. This is because of the fact that after eating this distance increases,
    but we don't want to prevent Pacman from eating food. This should only
    affect our decision if there is no food around to eat and we need to direct
    Pacman to the nearest food.
    To get this distance we use A-star search with Manhattan distance heuristic.
    We used simplified code from the previous assignment with deleted delivery points
    and modified goal - our goal now is just to pick item up.
    However, running A-star for each goodie on the map is too slow. That's why
    we use some sort of heuristic here before running A-star - we calculate manhattan
    distances to all goodies on the map, sort them according to this distance and then
    run A-star method only on some of them (5 closest in this version). Logic behind this is
    the following. We have some walls in our world and hence manhattan distance is
    not always gives us true estimate, but the points with lower manhattan distance
    have higher probability to be the closest. So, picking few of them using just
    manhattan distance has quite high chance to find the closest one (and at least it
    will give us <one of the closest> goodies).
    
    Our previous "submission.py" is now named "search.py" and previous "util.py" is
    named "searchutil.py".
    
    2. Second correction is the number of goodies - negative again, the more goodies,
    the lower the score. This encourages pacman to each everything he can. This one
    has higher coefficient of 1.
    
    3. Third correction is the number of capsules. We really want to eat them,
    so this has coefficient of 300. After the capsule is eaten, ghosts are scared
    and we include them in the list of goodies encouraging Pacman to eat them.
    
    ##### BERSERK MODE #####
    
    By setting <berserk = True> below you can force Pacman to forget about all other 
    food and chase only scared ghosts if there are any. This way Pacman usually
    gets above 1500-1700 points if he stays alive, but this leads to higher probability
    to get cornered by ghosts - often Pacman runs into their respawn zone and dies
    there if there is "not scared" ghost around.
    
    Berserk is off by default.
    
    
    All corrections are calculated by goodies() function. It returns the tuple
    of distance to the nearest goodie, number of goodies and number of capsules.
    
    
    Function create_search_problem() creates delivery problem for A-star search.
    
    
    This scoring function usually leads Pacman to win (10 out of 10) with average
    score 1500-1600 unless Pacman is very unlucky (usually when he dies, he dies
    because he gets cornered by two ghosts).
    
    
    """
    
    # BEGIN_YOUR_CODE (around 50 lines of code expected)
    # our base for the final score
    score = util.lookup('scoreEvaluationFunction', globals())(currentGameState)
    
    #### BERSERK MODE ####
    # let Pacman forget about capsules and food (at least if there is no food
    # in the immediate surroundings) and hunt the ghosts if there are scared
    # ghosts
    berserk = False # >_<
    
    #creates the map for the delivery problem to use with A*
    def create_search_problem(state, pacman, target):
        walls = state.getWalls()
        x = len(walls[:][:])
        y = len(walls[0][:])
        map = "\n"
        for j in reversed(range(y)):
            for i in range(x):
                if((i, j) == pacman):
                     map += "T "
                elif((i, j) == target):
                    map += "P0"
                elif(walls[i][j] == True):
                    map += "# "
                else:
                    map += ". "
                if(i != (x - 1)):
                    map += " "
            map = map[:-1]
            map += "\n"
        return map
    
    # described above
    def goodies(state, num):
        pacman = state.getPacmanPosition()
        food = state.getFood()
        capsules = state.getCapsules()
        x = len(food[:][:])
        y = len(food[0][:])
        scared = False
        l = [] # list of goodies - their coordinates and dist to Pacman: (x, y, dist)
        f = 0
        c = 0
        
        # if the ghost is scared, add it to the list
        for i in range(state.getNumAgents() - 1):
            if(state.getGhostStates()[i].scaredTimer > 0):
                scared = True
                gx = int(state.getGhostStates()[i].getPosition()[0])
                gy = int(state.getGhostStates()[i].getPosition()[1])
                dist = abs(pacman[0] - gx) + abs(pacman[1] - gy)
                f += 1
                l.append((gx, gy, dist))
        
        c = len(capsules)
        # add food and capsules to the list
        # here is the difference between usual and berserk modes
        if((not berserk) or not (berserk and scared)):
            for j in range(y):
                for i in range(x):
                    if(food[i][j] == True):
                        f += 1
                        l.append((i, j, abs(pacman[0] - i) + abs(pacman[1] - j)))
            for i in range(len(capsules)):
                l.append((capsules[i][0], capsules[i][1], abs(pacman[0] - capsules[i][0]) + abs(pacman[1] - capsules[i][1])))
        
        # if there is nothing to eat anymore (final stages of the game)
        if(len(l) == 0):
            return 0, 0, c
        
        # sort according to the manhattan distance
        l = sorted(l, key = lambda z: z[2])
        
        m = float("inf")
        
        # run A* for specified number of items (or less, if there is less items in the list)
        num = min(num, len(l))
        for i in range(num):
            scenario = create_search_problem(state, pacman, (l[i][0], l[i][1]))
            problem = search.DeliveryProblem(searchutil.DeliveryScenario(scenario))
            algorithm = search.AStarSearch(search.createHeuristic2(searchutil.DeliveryScenario(scenario)))
            algorithm.solve(problem)
            m = min(m, algorithm.totalCost)
        return m, f, c
    
    #dist to nearest goodie, number of food dots, number of capsules
    df, nf, nc = goodies(currentGameState, 5)
    
    # correct the score
    score -= 0.05 * df
    score -= 300  * nc
    score -= 1    * nf
    
    # this is just to force Python to print output
    sys.stdout.flush()
    return score
    # END_YOUR_CODE

# Abbreviation
better = betterEvaluationFunction


