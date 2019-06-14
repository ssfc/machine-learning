#    COMPUTECOST Compute cost for linear regression
#    J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
#    parameter for linear regression to fit the data points in X and y

import numpy as np

def computeCost(X, y, theta) :

    m = X.shape[0]; # number of training examples
    delta = np.dot(X, theta) - np.reshape(y, (m, 1));    
    J = np.dot(delta.T, delta) / (2*m);


    return J[0][0]
