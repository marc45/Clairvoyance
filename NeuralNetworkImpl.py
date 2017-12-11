import numpy as np
import matplotlib.pyplot as plt


def InitTheta(numInputs, HiddenLayers, numOutputs):
    theta = []

    for i in range(len(HiddenLayers)):
        if (i == 0):
            theta.append((np.random.rand(HiddenLayers[i], numInputs + 1) - 0.2) / numInputs) 
        else:
            theta.append((np.random.rand(HiddenLayers[i], HiddenLayers[i - 1] + 1) - 0.2) / HiddenLayers[i - 1])

    theta.append((np.random.rand(numOutputs, HiddenLayers[-1] + 1) - 0.2) / HiddenLayers[-1])

    return theta

def RunLogisticNeuralNetwork( X, Y, HiddenLayers, lam, iters, useRELU=False, printProgress=False):
    theta = InitTheta(X.shape[1], HiddenLayers, Y.shape[1])
    J = np.zeros(iters)

    for i in range(iters):
        if (i % 50 == 0) and (printProgress):
            print(str((i / iters) * 100) + '%')

        J[i], thetaD = NNCostFunction(X, Y, theta, lam, useRELU)

        for s in range(len(theta)):
            theta[s] = theta[s] - (0.005 * thetaD[s])
    return J, theta

# takes an ndarray and applies the sigmoid function to all elements
def Sigmoid( X ):
    return 1 / (1 + np.exp(-X))

def SigmoidGradient( X ):
    return np.multiply(Sigmoid(X), Sigmoid(1 - X))

def RELU( X ):
    print(np.sum(np.isnan(np.maximum(X, 0)) * 1))
    return np.maximum(X, 0)

def RELUGradient( X ):
    print(np.sum(np.isnan(X * (X > 0)) * 1))
    return X * (X > 0)

def CheckNNGradient(X, Y, Theta ):
    grad = []
    e = 0.0001

    for s in range(len(Theta)):
        print('checking Theta ' + str(s) + ', ' + str(Theta[s].shape))
        grad.append(np.zeros(Theta[s].shape))
        for i in range(Theta[s].shape[0]):
            for j in range(Theta[s].shape[1]):
                Theta[s][i, j] -= e
                L1 = NNCostNoGradient(X, Y, Theta, 0)
                Theta[s][i, j] += 2*e
                L2 = NNCostNoGradient(X, Y, Theta, 0)
                grad[s][i, j] = (L2 - L1) / (2 * e)
                Theta[s][i, j] -= e
    
    return grad


def NNCostNoGradient( X, Y, Theta, lam ):
    alpha = [np.append(np.ones((X.shape[0], 1)), X, axis=1).transpose()]
    z = [0]
    m = X.shape[0]

    #Feed forward
    for i in range(len(Theta)):
        z.append(np.matmul(Theta[i], alpha[i]))
        alpha.append(Sigmoid(z[i + 1]))
        if (i != len(Theta) - 1):
            alpha[i + 1] = np.append(np.ones((1, alpha[i + 1].shape[1])), alpha[i + 1], axis=0)

    for i in range(len(alpha)):
        alpha[i] = alpha[i].transpose()

    #computing cost
    Jmat = np.multiply(-Y, np.log(alpha[-1])) - np.multiply(1 - Y, np.log(1 - alpha[-1]))
    J = (1 / m) * np.sum(Jmat)
    for i in range(len(Theta)):
        J = J + ((lam / (2 * m)) * np.sum(np.power(Theta[i][:, 1:], 2)))

    return J



def NNCostFunction( X, Y, Theta, lam, useRELU ):
    alpha = [np.append(np.ones((X.shape[0], 1)), X, axis=1).transpose()]
    z = [0]
    m = X.shape[0]

    #Feed forward
    for i in range(len(Theta)):
        z.append(np.matmul(Theta[i], alpha[i]))
        if (useRELU):
            alpha.append(RELU(z[i + 1]))
        else:
            alpha.append(Sigmoid(z[i + 1]))
        if (i != len(Theta) - 1):
            alpha[i + 1] = np.append(np.ones((1, alpha[i + 1].shape[1])), alpha[i + 1], axis=0)

    for i in range(len(alpha)):
        alpha[i] = alpha[i].transpose()

    #back propegation
    delta = [alpha[-1] - Y]

    for i in range(len(alpha) - 1, 1, -1):
        if (useRELU):
            d = np.multiply(np.matmul(delta[0], Theta[i - 1])[:, 1:], RELUGradient(z[i - 1]).transpose())
        else:
            d = np.multiply(np.matmul(delta[0], Theta[i - 1])[:, 1:], SigmoidGradient(z[i - 1]).transpose())
        delta.insert(0, d)

    #computing cost
    Jmat = np.multiply(-Y, np.log(alpha[-1])) - np.multiply(1 - Y, np.log(1 - alpha[-1]))
    J = (1 / m) * np.sum(Jmat)
    for i in range(len(Theta)):
        J = J + ((lam / (2 * m)) * np.sum(np.power(Theta[i][:, 1:], 2)))

    ThetaGrad = []
    for i in range(len(Theta)):
        ThetaGrad.append((1 / m) * np.matmul(delta[i].transpose(), alpha[i]))
        ThetaGrad[i][:, 1:] = ThetaGrad[i][:, 1:] + ((lam / m) * Theta[i][:, 1:])
    
    return J, ThetaGrad
