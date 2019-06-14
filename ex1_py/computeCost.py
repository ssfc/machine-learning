import numpy as np

def computeCost(X, y, theta) :

    m = X.shape[0]; # number of training examples
    delta = np.dot(X, theta) - np.reshape(y, (m, 1));    
    J = np.dot(delta.T, delta) / (2*m);


    return J[0][0]
