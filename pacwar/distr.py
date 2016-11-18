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
    for i in range(len(a[0])):
        dd = [0, 0, 0, 0]
        for j in range(len(a)):
            dd[a[j][i]] += 1
        d.append(dd)
    return d
# Example Python module in C for Pacwar
def main():
    bests = np.zeros([1,50], dtype=int)
    bests[0] = generate_random()
    for i in range(1000):
        bests = np.vstack((bests, generate_random()))

if __name__ == "__main__": main()
