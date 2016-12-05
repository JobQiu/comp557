"""
Text classification
"""

import util
import operator
from collections import Counter
import numpy as np
import sys, copy, re

class Classifier(object):
    def __init__(self, labels):
        """
        @param (string, string): Pair of positive, negative labels
        @return string y: either the positive or negative label
        """
        self.labels = labels

    def classify(self, text):
        """
        @param string text: e.g. email
        @return double y: classification score; >= 0 if positive label
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, text):
        """
        @param string text: the text message
        @return string y: either 'ham' or 'spam'
        """
        if self.classify(text) >= 0.:
            return self.labels[0]
        else:
            return self.labels[1]

class RuleBasedClassifier(Classifier):
    def __init__(self, labels, blacklist, n=1, k=-1):
        """
        @param (string, string): Pair of positive, negative labels
        @param list string: Blacklisted words
        @param int n: threshold of blacklisted words before email marked spam
        @param int k: number of words in the blacklist to consider
        """
        super(RuleBasedClassifier, self).__init__(labels)
        #('ham', 'spam')
        # BEGIN_YOUR_CODE (around 3 lines of code expected)
        if k != -1:
            self.blacklist = Counter(blacklist[0:k])
        else:
            self.blacklist = Counter(blacklist)
        self.n = n
        # END_YOUR_CODE

    def classify(self, text):
        """
        @param string text: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 8 lines of code expected)
        score = self.n - 1
        for i in text.split(" "):
            if i in self.blacklist:
                score -= 1
        return score
        # END_YOUR_CODE

def extractUnigramFeatures(x):
    """
    Extract unigram features for a text document $x$. 
    @param string x: represents the contents of an text message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    x = x.replace('\n',' ')
    x = re.sub(r' +', ' ', x)
    return Counter(x.split(" "))
    # END_YOUR_CODE


class WeightedClassifier(Classifier):
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        super(WeightedClassifier, self).__init__(labels)
        self.featureFunction = featureFunction
        self.params = params

    def classify(self, x):
        """
        @param string x: the text message
        @return double y: classification score; >= 0 if positive label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        score = 0
        f = self.featureFunction(x)
        for i in f:
            if i in self.params:
                score += self.params[i] * f[i]
        return score
        # END_YOUR_CODE

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('pos', 'neg'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    # BEGIN_YOUR_CODE (around 15 lines of code expected)
    w = Counter({})
    for i in range(len(trainExamples)):
        for j in featureExtractor(trainExamples[i][0]):
            #print i, j
            #sys.stdout.flush()
            w[j] = 0
    
    for i in range(iters):
        #print i
        #sys.stdout.flush()
        for j in range(len(trainExamples)):
            #print i, j
            #sys.stdout.flush()
            c = WeightedClassifier(labels, featureExtractor, w)
            if c.classifyWithLabel(trainExamples[j][0]) != trainExamples[j][1]:
                d = featureExtractor(trainExamples[j][0])
                if(trainExamples[j][1] == labels[0]):
                    l = +1
                elif(trainExamples[j][1] == labels[1]):
                    l = -1
                else:
                    print labels, trainExamples[j][1]
                for k in d.keys():
                    w[k] += l * d[k]
    return w
    # END_YOUR_CODE

def extractBigramFeatures(x):
    """
    Extract unigram + bigram features for a text document $x$. 

    @param string x: represents the contents of an email message.
    @return dict: feature vector representation of x.
    """
    # BEGIN_YOUR_CODE (around 12 lines of code expected)
    #sym = ["!", "?", ",", ".", "\'", "\"", "--", " ", ")", "(", ":", ";", "-"]
    sym = []
    x = re.sub(r' +', ' ', x)
    t = x.split(" ")
    f = Counter(x.split(" "))
    f += Counter({str("-BEGIN-" + " " + t[0]):1})
    for i in range(len(t) - 1):
        if t[i] in ["!", "?", "."] and t[i + 1] not in sym:
            f[str("-BEGIN-" + " " + t[i + 1])] += 1
        elif t[i + 1] not in sym and t[i] not in sym:
            f[str(t[i] + " " + t[i + 1])] += 1
    return f
    # END_YOUR_CODE

class MultiClassClassifier(object):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); each classifier is a WeightedClassifier that detects label vs NOT-label
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        self.labels = labels
        self.classifiers = classifiers
        # END_YOUR_CODE

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        raise NotImplementedError("TODO: implement classify")

    def classifyWithLabel(self, x):
        """
        @param string x: the text message
        @return string y: one of the output labels
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        scores = [self.classifiers[i][1].classify(x) for i in range(len(self.labels))]
        return self.labels[np.argmax(scores)]
        # END_YOUR_CODE

class OneVsAllClassifier(MultiClassClassifier):
    def __init__(self, labels, classifiers):
        """
        @param list string: List of labels
        @param list (string, Classifier): tuple of (label, classifier); the classifier is the one-vs-all classifier
        """
        super(OneVsAllClassifier, self).__init__(labels, classifiers)

    def classify(self, x):
        """
        @param string x: the text message
        @return list (string, double): list of labels with scores 
        """
        # BEGIN_YOUR_CODE (around 4 lines of code expected)
        out = []
        for i in range(len(self.labels)):
            s = self.classifiers[i][1].classify(x)
            out.append((self.labels[i], s))
        return out
        # END_YOUR_CODE

def learnOneVsAllClassifiers(trainExamples, featureFunction, labels, perClassifierIters = 10):
    """
    Split the set of examples into one label vs all and train classifiers
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the text message, and
      - y is a string representing the label (an entry from the list of labels)
    @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
    @param list string labels: List of labels
    @param int perClassifierIters: number of iterations to train each classifier
    @return list (label, Classifier)
    """
    # BEGIN_YOUR_CODE (around 10 lines of code expected)
    out = []
    for i in range(len(labels)):
        train = []
        for j in range(len(trainExamples)):
            if trainExamples[j][1] == labels[i]:
                train.append((trainExamples[j][0], 'pos'))
            else:
                train.append((trainExamples[j][0], 'neg'))
                
        weights = learnWeightsFromPerceptron(train, featureFunction, ('pos', 'neg'), perClassifierIters)
        out.append((labels[i], WeightedClassifier(('pos', 'neg'), featureFunction, weights)))
    return out
    # END_YOUR_CODE

