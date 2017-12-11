import numpy as np
import matplotlib.pyplot as plt
import random
import math

def InitCentroids( X, numK ):
    K = np.zeros((numK, X.shape[1]))
    A = [i for i in range(X.shape[0])]
    random.shuffle(A)

    for i in range(numK):
        K[i, :] = X[A[i], :]

    return K

def FindClosestCentroid( X, K ):
    
    m = X.shape[0]
    idx = np.zeros((m, 1), dtype=int)
    n = np.ones((X.shape[1], 1))

    for i in range(m):
        idx[i, 0] = np.argmin(np.matmul(np.power(K - X[i, :], 2), n))
        
    return idx

def ComputeK( X, idx, numK ):
    minC = math.sqrt(X.shape[0] / numK)
    K = np.zeros((numK, X.shape[1]))
    c = np.zeros((numK, 1))

    for i in range(X.shape[0]):
        K[idx[i, 0], :] += X[i, :]
        c[idx[i, 0], 0] += 1

    empty = []
    for i in range(numK):
        if (c[i] >= minC):
            K[i, :] = K[i, :] * (1 / c[i])
        else:
            empty.append(i)
    
    mask = np.ones(K.shape[0], dtype=bool)
    mask[empty] = False

    return K[mask,...]

def NormalizeX( X ):
    n = np.sum(X, axis=0)
    c = np.count_nonzero(X, axis=0)

    #set all zeros in C to 1 to prevent divide by zero errors
    for i in range(c.shape[0]):
        if (c[i] == 0):
            c[i] = 1
        if (n[i] == 0):
            n[i] = 1

    X = X / (n / c)
    
    return X

def RunKMeansOnImage ( X, kGrid, kStep, iters ):
    numK = kGrid[0] * kGrid[1]
    K = np.zeros((numK, X.shape[1]))

    #this should only run on images, IE a test set with just two features (X, Y coord)
    if (X.shape[1] > 2):
        return K

    for i in range(K.shape[0]):
        K[i, 0] = ((i % kGrid[0]) * kStep) + (kStep / 2)
        K[i, 1] = ((i / kGrid[0]) * kStep) + (kStep / 2)


    for i in range(iters):
        idx = FindClosestCentroid(X, K)

        K = ComputeK(X, idx, K.shape[0])

    return K

def RunKMeans( X, numK, iters ):
    K = InitCentroids(X, numK)

    for i in range(iters):
        print (i)
        idx = FindClosestCentroid(X, K)
        K = ComputeK(X, idx, numK)

    return K