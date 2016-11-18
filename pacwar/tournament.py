import _PyPacwar, sys
import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd

# c2 is winner!
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
    d = [0.] * 4
    for i in range(len(d)):
        d[i] = [0.] * 50
    for i in range(len(a)):
        s = list(a[i][0])
        for j in range(len(s)):
            d[int(s[j])][j] += 1. / len(a)
    #print "\n"
    #for i in range(50):
    #    print i, "->>", d[0][i], d[1][i], d[2][i], d[3][i], "\n"
    return d

def plot_bests(a):
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)
    plt.figure(figsize=(16, 7))
    ax = plt.subplot(111)
    
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    plt.ylim(-0.1, 1.1)    
    plt.xlim(-1, 51)
    
    plt.yticks(np.arange(0, 1.1, 0.1), [str(x) for x in np.arange(0, 1.1, 0.1)], fontsize=14)
    plt.xticks(fontsize=14)
    
    for y in np.arange(0,1,0.1):
        plt.plot(range(0, 50), [y] * len(range(0, 50)), "--", lw=0.5, color="black", alpha=0.3)
        
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
    labelbottom="on", left="off", right="off", labelleft="on")
    labels = ['0', '1', '2', '3']
    ly_pos = [0.75, 0.65, 0.55, 0.45]
    
    #c_start = 5
    
    c_ind = [0, 4, 16, 6]
    
    plt.text(50, 0.9, "Gene value:  ", fontsize=14, color=tableau20[c_ind[0]])
    for i in range(len(a)):
        # Plot each line separately with its own color, using the Tableau 20
        # color set in order.
        #color = np.random.choice(20)
        color = c_ind[i]
        plt.plot(range(50), a[i], lw=1.5, color=tableau20[color])
        plt.text(53.5, ly_pos[i], labels[i], fontsize=14, color=tableau20[color])
            
    plt.text(27, 1.05, "Distribution of the gene values of the strongest Pac mites", fontsize=17, ha="center")
    plt.savefig("ga2_stage2.png", bbox_inches="tight")

# Example Python module in C for Pacwar
def main():
    bests = [line.rstrip('\n') for line in open('stage2_ga2')]
    res = []
    for i in range(len(bests)):
        res.append([bests[i], 0])
    for i in range(len(bests)):
        sys.stdout.write("\rIteration # %s" % str(i + 1))
        sys.stdout.flush()
        for j in range(len(bests)):
            if(i > j):
                (rounds, c1, c2) = _PyPacwar.battle(list(bests[i]), list(bests[j]))
                #print (rounds, c1, c2)
                if(c1 > c2):
                    (s2, s1) = score(rounds, c2, c1)
                    #print "won"
                    #print "c1", c1
                    #print "c2", c2
                    #print s1, s2
                else:
                    #print "lost"
                    (s1, s2) = score(rounds, c1, c2)
                    #print "c1", c1
                    #print "c2", c2
                    #print s1, s2
                #print "+", s1
                sys.stdout.flush()
                res[i][1] += s1
                res[j][1] += s2
                
    res = sorted(res, key = lambda z: z[1])
    
    plot_bests(get_distr(res[-100:]))
    print "\n"
    for i in range(len(res)):
        print res[-(i + 1)][0], res[-(i + 1)][1]
    print "\n"
    for i in range(200):
        print res[-(i + 1)][0], res[-(i + 1)][1]
    print "### THE WINNER ###"
    print res[-1][0]
    print "##################"
    #print get_distr(res[-2:])
    #res = get_distr(res)
    #print list(res[1][0])

if __name__ == "__main__": main()
