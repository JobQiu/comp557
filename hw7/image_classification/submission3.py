import collections
import numpy as np

############################################################
# Problem 3.1

def runKMeans(k,patches,maxIter):
    """
    Runs K-means to learn k centroids, for maxIter iterations.
    
    Args:
      k - number of centroids.
      patches - 2D numpy array of size patchSize x numPatches
      maxIter - number of iterations to run K-means for

    Returns:
      centroids - 2D numpy array of size patchSize x k
    """
    # This line starts you out with randomly initialized centroids in a matrix 
    # with patchSize rows and k columns. Each column is a centroid.
    centroids = np.random.randn(patches.shape[0],k)
   #print "Centroids:", centroids
    numPatches = patches.shape[1]

    for i in range(maxIter):
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
        
        # assignment step
        zee = np.zeros( (numPatches, k) )
        for p in range(numPatches):
            #print "p:", p
            temp = []
            for x in centroids.T:
                #print np.linalg.norm(patches[:,p] - x)
                temp.append(np.linalg.norm(patches[:,p] - x))
            #print np.argmin(temp)
            zee[p, np.argmin(temp)] = 1

        #? handle empty clusters by assigning the most distant point from the largest cluster to one of the empty clusters; do this until no more empty clusters
        while True:
            sums = np.sum(zee, axis=0)
           #print "zee:", zee
            if not (0 in sums):
               #print "no0zee"
                break
           #print "sums:", sums
            for c in range(len(sums)):
               #print "c:", c
                if (sums[c] == 0):
                    print "c == 0:", c
                    #find large cluster and most distant point to it
                    maxC = np.argmax(sums)
                   #print "maxC:", maxC
                    tempDistances, tempIndices = [], []
                    for p in range(numPatches):
                        if (zee[p, maxC] == 1):
                            tempDistances.append(np.linalg.norm(patches[:, p] - centroids[:, maxC]))
                            tempIndices.append(p)
                    maxP = tempIndices[np.argmax(tempDistances)]
                    zee[maxP, maxC] = 0
                    zee[maxP, c] = 1
                   #print "newZee:", zee
                    break

        # update step
        newC = np.zeros( (patches.shape[0], k) )
        for c in range(k):
            numPinC = 0
            for p in range(numPatches):
                if (zee[p, c]):
                    numPinC += 1
                    newC[:, c] += patches[:, p]
            #print "c:", c
            #print "newC:", newC[:, c]
            #print "numPinC:", numPinC
            newC[:, c] = newC[:, c] / numPinC

        centroids = newC.copy()
       #print "end of iter:", i, "max:", maxIter
        # END_YOUR_CODE

    return centroids

############################################################
# Problem 3.2

def extractFeatures(patches,centroids):
    """
    Given patches for an image and a set of centroids, extracts and return
    the features for that image.
    
    Args:
      patches - 2D numpy array of size patchSize x numPatches
      centroids - 2D numpy array of size patchSize x k
      
    Returns:
      features - 2D numpy array with new feature values for each patch
                 of the image in rows, size is numPatches x k
    """
    k = centroids.shape[1]
    numPatches = patches.shape[1]
    features = np.empty((numPatches,k))

    # BEGIN_YOUR_CODE (around 9 lines of code expected)
    raise "Not yet implemented"
    # END_YOUR_CODE
    return features

############################################################
# Problem 3.3.1

import math
def logisticGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of the logistic loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of logistic loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    raise "Not yet implemented."
    # END_YOUR_CODE

############################################################
# Problem 3.3.2
    
def hingeLossGradient(theta,featureVector,y):
    """
    Calculates and returns gradient of hinge loss function with
    respect to parameter vector theta.

    Args:
      theta - 1D numpy array of parameters
      featureVector - 1D numpy array of features for training example
      y - label in {0,1} for training example

    Returns:
      1D numpy array of gradient of hinge loss w.r.t. to theta
    """
    # BEGIN_YOUR_CODE (around 6 lines of code expected)
    raise "Not yet implemented."
    # END_YOUR_CODE

