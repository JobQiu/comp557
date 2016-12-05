import numpy as np
import itertools
import copy
import sys
 
print "METHOD                    TRAIN ACC      TEST ACC   "
print "----------------------------------------------------"

data = [line.rstrip('\n') for line in open('data/adult.data')]
for i in range(len(data)):
    data[i] = data[i].split(", ")

test = [line.rstrip('\n') for line in open('data/adult.test')]
for i in range(len(test)):
    test[i] = test[i][0:-1]
    test[i] = test[i].split(", ")

cat = [1, 3, 5, 6, 7, 8, 9, 13, 14]
data_coded = copy.deepcopy(data)
test_coded = copy.deepcopy(test)
code = ([[data[0][j]] for j in cat])
for i in range(len(data)):
    for j in cat:
        if data[i][j] in code[cat.index(j)]:
            data_coded[i][j] = code[cat.index(j)].index(data[i][j])
        else:
            code[cat.index(j)].append(data[i][j])
            data_coded[i][j] = len(code[cat.index(j)]) - 1
for i in range(len(test)):
    for j in cat:
        if test[i][j] in code[cat.index(j)]:
            test_coded[i][j] = code[cat.index(j)].index(test[i][j])
        else:
            print "Err: ", test[i][j], "is not in the list!"
X = np.array(data_coded, dtype = float)[:, 0:14]
X_test = np.array(test_coded, dtype = float)[:, 0:14]
y = np.array(data_coded, dtype = float)[:, 14]
y_test = np.array(test_coded, dtype = float)[:, 14]

### NAIVE BAYES ###
from sklearn.naive_bayes import GaussianNB

X_nb = copy.deepcopy(X)
X_test_nb = copy.deepcopy(X_test)

def normalize(d):
    min = []
    max = []
    for i in range(len(d[0])):
        max.append(np.max(d[:, i]))
        min.append(np.min(d[:, i]))
        
    for i in range(len(d)):
        for j in range(len(d[i])):
            if (max[j] - min[j] != 0):
                d[i][j] = (d[i][j] - min[j]) / (max[j] - min[j])
            else:
                d[i][j] = (d[i][j] - min[j])
    return d


def NBify(d):
    for i in range(len(d)):
        for j in range(15):
            if j == 0: #age: 17 - 25; 26 - 35; 36 - 45; 46 - 60; 61 - 90
                if d[i][j] < 26:
                    d[i][j] = 0
                elif d[i][j] < 36:
                    d[i][j] = 1
                elif d[i][j] < 46:
                    d[i][j] = 2
                elif d[i][j] < 61:
                    d[i][j] = 3
                else:
                    d[i][j] = 4
            elif j == 2: # fnlwgt: 12K - 500K; 500K - 1000K; 1000K - 1500K
                if d[i][j] < 500001:
                    d[i][j] = 0
                elif d[i][j] < 1000001:
                    d[i][j] = 1
                else:
                    d[i][j] = 2
            elif j == 4: # ed_num: 1 - 3; 4 - 6; 7 - 9; 10 - 12; 13 - 16
                if d[i][j] < 4:
                    d[i][j] = 0
                elif d[i][j] < 7:
                    d[i][j] = 1
                elif d[i][j] < 10:
                    d[i][j] = 2
                elif d[i][j] < 13:
                    d[i][j] = 3
                else:
                    d[i][j] = 4
            elif j == 10: # capital-gain: 0 - 25K; 25K - 50K; 50K - 75K; 75K - 100K
                if d[i][j] < 25001:
                    d[i][j] = 0
                elif d[i][j] < 50001:
                    d[i][j] = 1
                elif d[i][j] < 75001:
                    d[i][j] = 2
                else:
                    d[i][j] = 3
            elif j == 11: # capital-loss: 0 - 1K; 1K - 2K; 2K - 3K; 3K - more
                if d[i][j] < 1001:
                    d[i][j] = 0
                elif d[i][j] < 2001:
                    d[i][j] = 1
                elif d[i][j] < 3001:
                    d[i][j] = 2
                else:
                    d[i][j] = 3
            elif j == 12: # HPW: 0 - 25; 26 - 50; 51 - 75; 76 - 99
                if d[i][j] < 26:
                    d[i][j] = 0
                elif d[i][j] < 51:
                    d[i][j] = 1
                elif d[i][j] < 76:
                    d[i][j] = 2
                else:
                    d[i][j] = 3
    return d

#X_nb = NBify(X_nb)
#X_test_nb = NBify(X_test_nb)
nb = GaussianNB()
nb.fit(X_nb, y)
print "NAIVE BAYES:          ", np.average(nb.predict(X_nb) == y), np.average(nb.predict(X_test_nb) == y_test)

### PERCEPTRON ###
from sklearn.linear_model import Perceptron
pct = Perceptron()
X_pct = copy.deepcopy(X)
X_test_pct = copy.deepcopy(X_test)
X_pct = np.insert(X_pct, 0, 1, axis=1)
X_test_pct = np.insert(X_test_pct, 0, 1, axis=1)
pct.fit(X_pct, y)
print "PERCEPTRON:           ", np.average(pct.predict(X_pct) == y), np.average(pct.predict(X_test_pct) == y_test)

X_pct = normalize(copy.deepcopy(X))
X_test_pct = normalize(copy.deepcopy(X_test))
X_pct = np.insert(X_pct, 0, 1, axis=1)
X_test_pct = np.insert(X_test_pct, 0, 1, axis=1)
#X_pct = NBify(X)
#X_test_pct = NBify(X_test)
pct.fit(X_pct, y)
print "PERCEPTRON NORM:      ", np.average(pct.predict(X_pct) == y), np.average(pct.predict(X_test_pct) == y_test)

### DECISION TREES ###
from sklearn import tree
tr = tree.DecisionTreeClassifier()
tr.fit(X, y)
print "TREES:                ", np.average(tr.predict(X) == y), np.average(tr.predict(X_test) == y_test)

### NEURAL NETWORK ###
from sklearn.neural_network import MLPClassifier

X_net = X
X_test_net = X_test
X_net = np.insert(copy.deepcopy(X_net), 0, 1, axis=1)
X_test_net = np.insert(copy.deepcopy(X_test_net), 0, 1, axis=1)
net = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(100, 100), random_state=1)
net.fit(X_net, y)
print "NEURAL NET:           ", np.average(net.predict(X_net) == y), np.average(net.predict(X_test_net) == y_test)

X_net = normalize(X)
X_test_net = normalize(X_test)
X_net = np.insert(copy.deepcopy(X_net), 0, 1, axis=1)
X_test_net = np.insert(copy.deepcopy(X_test_net), 0, 1, axis=1)

"""

arr = [5, 10, 15, 20, 50]
train_scores = np.zeros((5, 5))
test_scores = np.zeros((5, 5))

for h1 in range(len(arr)):
    for h2 in range(len(arr)):
        net = MLPClassifier(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(arr[h1], arr[h2]), random_state=1)
        
        r = np.random.choice(range(len(X_net)), int(len(X_net) * 0.8), replace = False)
        net.fit([X_net[i] for i in range(len(X_net)) if i in r], [y[i] for i in range(len(X_net)) if i in r])
        s = net.predict([X_net[i] for i in range(len(X_net)) if i not in r]) == [y[i] for i in range(len(X_net)) if i not in r]
        train_scores[h1, h2] = np.average(net.predict([X_net[i] for i in range(len(X_net)) if i in r]) == [y[i] for i in range(len(X_net)) if i in r])
        test_scores[h1, h2] = np.average(s)
        print arr[h1], arr[h2], "->", train_scores[h1, h2], test_scores[h1, h2]
        sys.stdout.flush()

print train_scores
print test_scores

"""
net = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(100, 100), random_state=1)
net.fit(X_net, y)
print "NEURAL NET NORM RELU: ", np.average(net.predict(X_net) == y), np.average(net.predict(X_test_net) == y_test)
"""
net = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(100, 100), random_state=1, activation = 'tanh')
net.fit(X_net, y)
print "NEURAL NET NORM TANH: ", np.average(net.predict(X_net) == y), np.average(net.predict(X_test_net) == y_test)

net = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(100, 100), random_state=1, activation = 'logistic')
net.fit(X_net, y)
print "NEURAL NET NORM SIGM: ", np.average(net.predict(X_net) == y), np.average(net.predict(X_test_net) == y_test)

net = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(100, 100), random_state=1, activation = 'identity')
net.fit(X_net, y)
print "NEURAL NET NORM IDEN: ", np.average(net.predict(X_net) == y), np.average(net.predict(X_test_net) == y_test)
"""
