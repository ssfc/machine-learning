
import numpy as np

# import user defined functions;
import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
#    GRADIENTDESCENT Performs gradient descent to learn theta
#    theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
#    taking num_iters gradient steps with learning rate alpha

#    Initialize some useful values

    m = X.shape[0]; # number of training examples
    J_history = np.zeros([num_iters, 1]);

    for iter in range(0, num_iters-1):
        
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta. 
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCost) and gradient here.
        #
        
        delta = np.dot(X, theta) - np.reshape(y, (m, 1)) # 97*1         
        theta = theta - (alpha/m) * np.dot(X.T, delta)        
        
        # ============================================================

        # Save the cost J in every iteration    
        J_history[iter][0] = computeCost.computeCost(X, y, theta);
        

    return theta