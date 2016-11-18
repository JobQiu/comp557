import _PyPacwar, sys
import numpy as np

P=np.array([[0.25, 0.25, 0.25, 0.25],         # 0 not used
            [0.25, 0.25, 0.25, 0.25],         # 1 not used
            [0.25, 0.25, 0.25, 0.25],         # 2 not used
            [0.25, 0.25, 0.25, 0.25],         # 3 not used
            [0.7, 0.1, 0.1, 0.1],             # 4
            [0.7, 0.1, 0.1, 0.1],             # 5
            [0.7, 0.1, 0.1, 0.1],             # 6
            [0.7, 0.1, 0.1, 0.1],             # 7
            [0.4, 0.4, 0.1, 0.1],             # 8
            [0.4, 0.4, 0.1, 0.1],             # 9
            [0.4, 0.4, 0.1, 0.1],             # 10
            [0.4, 0.4, 0.1, 0.1],             # 11
            [0.4, 0.1, 0.4, 0.1],             # 12
            [0.4, 0.1, 0.4, 0.1],             # 13
            [0.4, 0.1, 0.4, 0.1],             # 14
            [0.4, 0.1, 0.4, 0.1],             # 15
            [0.4, 0.1, 0.1, 0.4],             # 16
            [0.4, 0.1, 0.1, 0.4],             # 17
            [0.4, 0.1, 0.1, 0.4],             # 18
            [0.4, 0.1, 0.1, 0.4],             # 19
            [0.25, 0.25, 0.25, 0.25],         # 20 not used
            [0.25, 0.25, 0.25, 0.25],         # 21 not used
            [0.25, 0.25, 0.25, 0.25],         # 22 not used
            [0.25, 0.25, 0.25, 0.25],         # 23 not used
            [0.25, 0.25, 0.25, 0.25],         # 24 not used
            [0.25, 0.25, 0.25, 0.25],         # 25 not used
            [0.1, 0.3, 0.3, 0.3],             # 26
            [0.1, 0.3, 0.3, 0.3],             # 27
            [0.1, 0.3, 0.3, 0.3],             # 28
            [0.1, 0.3, 0.3, 0.3],             # 29
            [0.1, 0.3, 0.3, 0.3],             # 30
            [0.1, 0.3, 0.3, 0.3],             # 31
            [0.1, 0.3, 0.3, 0.3],             # 32
            [0.1, 0.3, 0.3, 0.3],             # 33
            [0.1, 0.3, 0.3, 0.3],             # 34
            [0.1, 0.3, 0.3, 0.3],             # 35
            [0.1, 0.3, 0.3, 0.3],             # 36
            [0.1, 0.3, 0.3, 0.3],             # 37
            [0.1, 0.3, 0.3, 0.3],             # 38
            [0.1, 0.3, 0.3, 0.3],             # 39
            [0.1, 0.3, 0.3, 0.3],             # 40
            [0.1, 0.3, 0.3, 0.3],             # 41
            [0.1, 0.3, 0.3, 0.3],             # 42
            [0.1, 0.3, 0.3, 0.3],             # 43
            [0.1, 0.3, 0.3, 0.3],             # 44
            [0.1, 0.3, 0.3, 0.3],             # 45
            [0.1, 0.3, 0.3, 0.3],             # 46
            [0.1, 0.3, 0.3, 0.3],             # 47
            [0.1, 0.3, 0.3, 0.3],             # 48
            [0.1, 0.3, 0.3, 0.3]])            # 49


def mix(parents):
    positions=np.array([[0, 4],
                        [4, 8],
                        [8, 12],
                        [12, 16],
                        [16, 20],
                        [20, 23],
                        [23, 26],
                        [26, 29],
                        [29, 32],
                        [32, 35],
                        [35, 38],
                        [38, 41],
                        [41, 44],
                        [44, 47],
                        [47, 50]])
    out = []
    for n in range(0):
        g = ''
        for i in range(15):
            r = np.random.choice(len(parents))
            g += parents[r][positions[i][0]:positions[i][1]]
        out.append(g)
    for n in range(1000):
        g = ''
        for i in range(15):
            r = np.random.choice(len(parents))
            g += parents[r][positions[i][0]:positions[i][1]]
        g = mutate(g, np.random.choice(range(1, 7)))
        out.append("".join(str(x) for x in g))
    #print "out"
    #print out
    return out


def score(r, c1, c2):
    if(r < 100):
        return 0, 20
    elif(r < 199):
        return 1, 19
    elif(r < 299):
        return 2, 18
    elif(r < 500 or (r == 500 and c1 != c2)):
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
        if(np.random.choice([0,1], p=[0.5,0.5]) == 0):
            c = np.random.choice(range(4, 20))
        else:
            c = np.random.choice(range(26, 50))
        new[c] = np.random.choice([0, 1, 2, 3], p=P[c])
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
    for i in range(5):
        parents = [line.rstrip('\n') for line in open('0131')]
        bests = [line.rstrip('\n') for line in open('stage2_ga3')]
        new_best = []
        for i in range(100):
            sys.stdout.write("\r                                  ")
            res = []
            offsprings = mix(parents)
            for m in range(len(offsprings)):
                res.append([offsprings[m], 0])
            for j in range(len(offsprings)):
                sys.stdout.write("\rIteration # %s - %s" % (str(i + 1), str(j + 1)))
                sys.stdout.flush()
                for k in range(len(bests)):
                    (rounds, c1, c2) = _PyPacwar.battle(list(bests[k]), list(offsprings[j]))
                    if(c1 > c2):
                        (s2, s1) = score(rounds, c2, c1)
                    else:
                        (s1, s2) = score(rounds, c1, c2)
                    res[j][1] += s2
            res = sorted(res, key = lambda z: z[1])
            #print "res"
            #print res
            #print "\n"
            if res[-1][0] not in new_best:
                new_best.append(res[-1][0])
                #print "best", res[-1][0]
            #else:
                #print "best", res[-1][0], " - duplicate"
        print "\n"
        print "### NEW BEST GENES ###"
        for i in range(len(new_best)):
            print new_best[i]
if __name__ == "__main__": main()
