import sys, operator
import util, submission2
import numpy as np
import matplotlib.pyplot as plt

# Main entry point to test your spam classifier.

TRAIN_PATH_SPAM = 'data/spam-classification/train'
TRAIN_PATH_SENTIMENT = 'data/sentiment/train'
TRAIN_PATH_TOPICS = 'data/topics/train'

def plot(a):
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
    
    plt.ylim(-0.1, 110)
    plt.xlim(0.5, 21)
    
    plt.yticks(np.arange(0, 110, 10), [str(x) for x in np.arange(0, 110, 10)], fontsize=14)
    plt.xticks(np.arange(1, 21, 1), [str(x) for x in np.arange(1, 21, 1)], fontsize=14)
    
    for y in np.arange(0, 110, 10):
        plt.plot(range(1, 21), [y] * len(range(0, 20)), "--", lw=0.5, color="black", alpha=0.3)
        
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
    labelbottom="on", left="off", right="off", labelleft="on")
    labels = [' - with punctuation', ' - without punctuation']
    ly_pos = [62, 52]
    
    #c_start = 5
    
    #c_ind = [0, 4, 16, 6]
    c_ind = [0, 4]
    
    #plt.text(21, 95, "Dev err:  ", fontsize=14, color=tableau20[c_ind[0]])
    for i in range(len(a)):
        # Plot each line separately with its own color, using the Tableau 20
        # color set in order.
        #color = np.random.choice(20)
        color = c_ind[i]
        plt.plot(range(1, 21, 1), a[i], lw=1.5, color=tableau20[color])
        plt.text(16.1, ly_pos[i], labels[i], fontsize=14, color=tableau20[color])
            
    plt.text(10, 105, "Test errors of classification of positive and negative reviews", fontsize=17, ha="center")
    plt.text(10, -10, "Number of iterations", fontsize=12, ha="center")
    plt.savefig("dev.png", bbox_inches="tight")

def evaluateClassifier(trainExamples, devExamples, classifier):
    #util.printConfusionMatrix(util.computeConfusionMatrix(trainExamples, classifier))
    trainErrorRate = util.computeErrorRate(trainExamples, classifier) 
    print 'trainErrorRate: %f' % trainErrorRate
    #util.printConfusionMatrix(util.computeConfusionMatrix(devExamples, classifier))
    devErrorRate = util.computeErrorRate(devExamples, classifier) 
    print 'devErrorRate: %f' % devErrorRate
    return (trainErrorRate, devErrorRate)

def part1_1(args):
    print "Part 2.1.1 RuleBasedClassifier"

    examples = util.loadExamples(TRAIN_PATH_SPAM)[:args.examples]
    labels = util.LABELS_SPAM
    trainExamples, devExamples = util.holdoutExamples(examples)
    classifier = submission2.RuleBasedClassifier( 
            labels, util.loadBlacklist(), args.n, args.k)
    
    dev = np.zeros((3, 3))
    n = [1, 2, 3]
    k = [10000, 20000, 30000]
    for i in range(len(n)):
        for j in range(len(k)):
            print "n: ", n[i], "k: ", k[j]
            classifier = submission2.RuleBasedClassifier(labels, util.loadBlacklist(), n[i], k[j])
            (tr, dev[i][j]) = evaluateClassifier(trainExamples, devExamples, classifier)
    print dev

def part1_3(args):
    print "Part 2.1.3 learnWeightsFromPerceptron"

    examples = util.loadExamples(TRAIN_PATH_SPAM)[:args.examples]
    labels = util.LABELS_SPAM
    trainExamples, devExamples = util.holdoutExamples(examples)
    weights = submission2.learnWeightsFromPerceptron(trainExamples, submission2.extractUnigramFeatures, labels, args.iters)
    classifier = submission2.WeightedClassifier(labels, submission2.extractUnigramFeatures, weights)

    evaluateClassifier(trainExamples, devExamples, classifier)

    #print "The bigram feature extractor"
    #print submission2.extractBigramFeatures('The quick dog chased the lazy fox over the brown fence.')

    print "Varying the number of examples with unigram features:"
    tr = []
    dev = []
    for i in range(500,5500,500):
        weights = submission2.learnWeightsFromPerceptron(trainExamples[:i], submission2.extractUnigramFeatures, labels, args.iters)
        classifier = submission2.WeightedClassifier(labels, submission2.extractUnigramFeatures, weights)
        
        (t, d) = evaluateClassifier(trainExamples[:i], devExamples, classifier)
        tr.append(t * 100)
        dev.append(d * 100)
    print "tr: ", tr
    print "dev: ", dev
    # output for latex
    #print "%.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f" % (dev[0], dev[1], dev[2], dev[3], dev[4], dev[5], dev[6], dev[7], dev[8], dev[9])
    #print "%.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f" % (tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], tr[6], tr[7], tr[8], tr[9])

    print "Varying the number of examples with bigram features:"
    tr = []
    dev = []
    for i in range(500,5500,500):
        weights = submission2.learnWeightsFromPerceptron(trainExamples[:i], submission2.extractBigramFeatures, labels, args.iters)
        classifier = submission2.WeightedClassifier(labels, submission2.extractBigramFeatures, weights)
        
        (t, d) = evaluateClassifier(trainExamples[:i], devExamples, classifier)
        tr.append(t * 100)
        dev.append(d * 100)
    
    # output for latex
    #print "%.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f" % (dev[0], dev[1], dev[2], dev[3], dev[4], dev[5], dev[6], dev[7], dev[8], dev[9])
    #print "%.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f" % (tr[0], tr[1], tr[2], tr[3], tr[4], tr[5], tr[6], tr[7], tr[8], tr[9])

def part2(args):
    print "Part 2.2 Sentiment Analysis"
    #sys.stdout.flush()
    #examples = util.loadExamples(TRAIN_PATH_SENTIMENT)[:args.examples]
    #labels = util.LABELS_SENTIMENT
    #trainExamples, devExamples = util.holdoutExamples(examples)
    #
    #weights = submission2.learnWeightsFromPerceptron(trainExamples, submission2.extractUnigramFeatures, labels, args.iters)
    #classifier = submission2.WeightedClassifier(labels, submission2.extractUnigramFeatures, weights)
    #
    #weights = submission2.learnWeightsFromPerceptron(trainExamples, submission2.extractBigramFeatures, labels, args.iters)
    #classifier = submission2.WeightedClassifier(labels, submission2.extractBigramFeatures, weights)
    #
    #evaluateClassifier(trainExamples, devExamples, classifier)
    #
    #tr = [0] * 20
    #dev = [0] * 20
    #
    #for i in range(1,21):
    #    print "Iters = ",i
    #    sys.stdout.flush()
    #    weights = submission2.learnWeightsFromPerceptron(trainExamples, submission2.extractBigramFeatures, labels, i)
    #    classifier = submission2.WeightedClassifier(labels, submission2.extractBigramFeatures, weights)
    #    (tr[i - 1], dev[i - 1]) = evaluateClassifier(trainExamples, devExamples, classifier)
    #print "tr", tr
    #print "dev", dev
    
    ## no punctuation
    tr1 = [0.5024330900243309, 0.46472019464720193, 0.09002433090024331, 0.2639902676399027,
    0.049878345498783457, 0.08029197080291971, 0.07907542579075426, 0.024330900243309004,
    0.5, 0.031630170316301706, 0.01338199513381995, 0.49391727493917276, 0.010948905109489052,
    0.009732360097323601, 0.004866180048661801, 0.004866180048661801, 0.0012165450121654502,
    0.004866180048661801, 0.0, 0.0]
    dev1 = [0.5112359550561798, 0.5, 0.21348314606741572, 0.398876404494382, 0.19101123595505617,
    0.2808988764044944, 0.2247191011235955, 0.15168539325842698, 0.5112359550561798, 0.2247191011235955,
    0.17415730337078653, 0.5056179775280899, 0.1853932584269663, 0.15168539325842698, 0.15730337078651685,
    0.15730337078651685, 0.17415730337078653, 0.19662921348314608, 0.1797752808988764, 0.1797752808988764]
    # punctuation
    tr2 = [0.47566909975669097, 0.5012165450121655, 0.09245742092457421, 0.058394160583941604,
    0.2664233576642336, 0.49635036496350365, 0.0705596107055961, 0.012165450121654502,
    0.025547445255474453, 0.03527980535279805, 0.045012165450121655, 0.01338199513381995,
    0.004866180048661801, 0.004866180048661801, 0.47566909975669097, 0.0, 0.0, 0.0, 0.0, 0.0]
    dev2 = [0.5, 0.5112359550561798, 0.25842696629213485, 0.19662921348314608, 0.398876404494382,
    0.5056179775280899, 0.2303370786516854, 0.17415730337078653, 0.19101123595505617, 0.19662921348314608,
    0.1853932584269663, 0.15730337078651685, 0.16853932584269662, 0.1797752808988764, 0.5,
    0.16292134831460675, 0.16292134831460675, 0.16292134831460675, 0.16292134831460675, 0.16292134831460675]
    
    plot([[dev1[i] * 100 for i in range(len(dev1))], [dev2[i] * 100 for i in range(len(dev2))]])
    for i in range(20):
        print "%d & %.2f & %.2f, %.2f, %.2f" % (i + 1, tr1[i], dev1[i], tr2[i], dev2[i])
    
        
def part3(args):
    print "Part 2.3 Topic Classification"
    examples = util.loadExamples(TRAIN_PATH_TOPICS)[:args.examples]
    labels = util.LABELS_TOPICS
    trainExamples, devExamples = util.holdoutExamples(examples)

    classifiers = submission2.learnOneVsAllClassifiers(trainExamples, submission2.extractUnigramFeatures, labels, 20)
    classifier = submission2.OneVsAllClassifier(labels, classifiers)

    evaluateClassifier(trainExamples, devExamples, classifier)
   

def main():
    import argparse
    parser = argparse.ArgumentParser( description='Spam classifier' )
    parser.add_argument('--examples', type=int, default=10000, help="Maximum number of examples to use" )
    subparsers = parser.add_subparsers()

    # Part 2.1.1
    parser1_1 = subparsers.add_parser('part2.1.1', help = "Part 2.1.1")
    parser1_1.add_argument('-n', type=int, default="1", help="Number of words to consider" )
    parser1_1.add_argument( '-k', type=int, default="-1", help="Number of words in blacklist to choose" )
    parser1_1.set_defaults(func=part1_1)

    # Part 2.1.3
    parser1_3 = subparsers.add_parser('part2.1.3', help = "Part 2.1.3")
    parser1_3.add_argument('--iters', type=int, default="20", help="Number of iterations to run perceptron" )
    parser1_3.set_defaults(func=part1_3)

    # Part 2.2
    parser2 = subparsers.add_parser('part2.2', help = "Part 2.2")
    parser2.add_argument('--iters', type=int, default="20", help="Number of iterations to run perceptron" )
    parser2.set_defaults(func=part2)

    # Part 2.3
    parser3 = subparsers.add_parser('part2.3', help = "Part 2.3")
    parser3.add_argument('--iters', type=int, default="20", help="Number of iterations to run perceptron" )
    parser3.set_defaults(func=part3)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

