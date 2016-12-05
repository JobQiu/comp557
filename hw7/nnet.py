from sklearn.neural_network import MLPClassifier
import numpy as np
import itertools

X = [[1, 0, 0, 0],
     [1, 0.5, 0, 1],
     [1, 0, 1, 0],
     [1, 1, 0, 1],
     [1, 0.5, 1, 0],
     [1, 1, 1, 1],
     [1, 0, 0, 0],
     [1, 1, 0, 0],
     [1, 0, 1, 1],
     [1, 1, 1, 0]]
y = [1, -1, 1, 1, -1, 1, 1, -1, 1, -1]
x_test= [[1, 1, 0, 0],
         [1, 1, 0, 1],
         [1, 0.5, 1, 1]]


for h in range(2, 10):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(h), random_state=1)
    scores = []
    for i in itertools.combinations(range(0, 10), 2):
        clf.fit([X[j] for j in range(0, 10) if j not in i], [y[j] for j in range(0, 10) if j not in i])
        scores.append(clf.score([X[j] for j in range(0, 10) if j in i], [y[j] for j in range(0, 10) if j in i]))
    print h, "->", np.average(scores)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(6), random_state=1)
clf.fit(X, y)
"""
MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
"""
print clf.predict(x_test)