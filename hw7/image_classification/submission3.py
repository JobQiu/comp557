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

    numPatches = patches.shape[1]

    for i in range(maxIter):
        # BEGIN_YOUR_CODE (around 19 lines of code expected)
        
        # assignment step
        zee = np.zeros( (numPatches, k) )
        for p in range(numPatches):
            tempDistances = []
            for x in centroids.T:
                #tempDistances.append(math.sqrt(sum([(a-b)**2 for a,b in zip(patches[:,p], x)])))
                tempDistances.append(np.linalg.norm(patches[:,p] - x))
            zee[p, np.argmin(tempDistances)] = 1

        # handle empty clusters by assigning the most distant point from the largest cluster to one of the empty clusters; do this until no more empty clusters
        while True:
            sums = np.sum(zee, axis=0)
            if not (0 in sums):
                break
            for c in range(len(sums)):
                if (sums[c] == 0):
                    maxC = np.argmax(sums)
                    tempDistances, tempIndices = [], []
                    for p in range(numPatches):
                        if (zee[p, maxC] == 1):
                            tempDistances.append(np.linalg.norm(patches[:, p] - centroids[:, maxC]))
                            tempIndices.append(p)
                    maxP = tempIndices[np.argmax(tempDistances)]
                    zee[maxP, maxC] = 0
                    zee[maxP, c] = 1
                    break

        # update step
        newC = np.zeros( (patches.shape[0], k) )
        for c in range(k):
            numPinC = 0
            for p in range(numPatches):
                if (zee[p, c]):
                    numPinC += 1
                    newC[:, c] += patches[:, p]
            newC[:, c] = newC[:, c] / numPinC

        centroids = newC.copy()

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

    distances = np.zeros( (numPatches, k) )
    for p in range(numPatches):
        for c in range(k):
            distances[p, c] = np.linalg.norm(patches[:, p] - centroids[:, c])

    for p in range(numPatches):
        for c in range(k):
            features[p, c] = max(np.mean(distances, axis=1)[p] - distances[p, c], 0)

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

    x = featureVector * (2 * y - 1)
    return -(x / (math.exp(np.dot(theta, x)) + 1))

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

    loss = np.zeros(featureVector.size)
    v = np.dot(theta, featureVector) * (2 * y - 1)

    for f in range(featureVector.size):
        if (v < 1):
            loss[f] = -(featureVector[f] * (2 * y - 1))

    return loss

    # END_YOUR_CODE

