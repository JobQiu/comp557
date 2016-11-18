import _PyPacwar, sys
import numpy as np

def score(r, c1, c2):
    if(r < 100):
        return 0, 20
    elif(r < 199):
        return 1, 19
    elif(r < 299):
        return 2, 18
    elif(r < 500):
        return 3, 17
    else:
        if(c2 / c1 > 10):
            return 7, 13
        elif(c2 / c1 > 3):
            return 8, 12
        elif(c2 / c1 > 1.5):
            return 9, 11
        else:
            return 10, 10
def eat_gene(g):
    return list(g)
    
def mutate(s, n):
    new = list(s)
    #print "\nMutating this -> ", s
    for i in range(n):
        new[np.random.choice(len(new))] = np.random.choice([0, 1, 2, 3])
    return new

def generate_random():
    g = []
    for i in range(50):
        g.append(np.random.choice([0, 1, 2, 3]))
    return g
def get_distr(a):
    d = []
    for i in range(len(a)):
        dd = [0, 0, 0, 0]
        for j in range(len(a[0])):
            dd[a[i][j]] += 1
        d[i] = dd
    return d
# Example Python module in C for Pacwar
def main():
    
    for i in range(10):
        bests = np.zeros([1,50], dtype=int)
        bests[0] = generate_random()
        #bests[0] = [3]*50
        N = 1000000
        for i in range(N):
            sys.stdout.write("\rIteration # %s" % str(i + 1))
            sys.stdout.flush()
            loser = False
            new = mutate(list(bests[len(bests) - 1]), 30)
            for j in range(len(bests)):
                (rounds, c1, c2) = _PyPacwar.battle(list(bests[j]), new)
                if(c1 > c2):
                    loser = True
                    break
            if(not loser):
                bests = np.vstack((bests, new))
                #print "\n"
                #print "".join(str(x) for x in new)
                loser = False
        print "\n"
        print "### BEST GENES ###"
        sys.stdout.flush()
        for i in range(10):
            print "".join(map(str, bests[-(i + 1)]))
            sys.stdout.flush()

if __name__ == "__main__": main()
