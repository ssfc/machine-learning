import matplotlib.pyplot as plt  # 绘图库
import numpy as np
import scipy.optimize as op

# import user defined functions;
import plotData


def sigmoid(z):
    return 1/(1 + np.exp(-z))


def computeGradient(theta, X, y):
    m, n = X.shape
    theta = theta.reshape((n,1))
    
    hypothesis = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, sigmoid(hypothesis - y)) / m;

    return grad


def computeCost(theta, X, y):
    m = X.shape[0]

    hypothesis = sigmoid(np.dot(X, theta))
    J = (np.dot(y.T, np.log(hypothesis)) + np.dot((1 - y).T, np.log(1-hypothesis))) / (-m)

    return J


def predict(theta, X):
    m = X.shape[0]    # Number of training examples

#    You need to return the following variables correctly
    p = np.zeros([m, 1])
    probability = np.zeros([m, 1])

    for i in range(0, m-1):
        probability[i][0] = sigmoid(np.dot(X[i,:], theta))
    
        if probability[i][0] >= 0.5:
            p[i][0] = 1
        else:
            p[i][0] = 0

    return p


data = np.loadtxt('ex2data1.txt')
X = data[:, [0, 1]]

y = data[:, 2]
y = y.reshape((X.shape[0],1))

## ==================== Part 1: Plotting ====================
print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")

#plotData.plotData(X, y)



## ============ Part 2: Compute Cost and Gradient ============
X = np.c_[np.ones([X.shape[0], 1]), X]    # Add a column of ones to x
m, n = X.shape
initial_theta = np.zeros([n, 1])

# Compute and display initial cost and gradient
cost = computeCost(initial_theta, X, y)
grad = computeGradient(initial_theta, X, y)

print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros):')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.zeros([3, 1])
test_theta[0][0] = -24
test_theta[1][0] = 0.2
test_theta[2][0] = 0.2
cost = computeCost(test_theta, X, y)
grad = computeGradient(test_theta, X, y)

print('Cost at test theta:', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta:')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')


## ============= Part 3: Optimizing using fminunc  =============
Result = op.minimize(fun = computeCost, 
                                 x0 = initial_theta, 
                                 args = (X, y),
                                 method = 'TNC',
                                 jac = computeGradient)
theta = Result.x

# Print theta to screen
print('Cost at theta found by fminunc: ', cost);
print('Expected cost (approx): 0.203');
print('theta:');
print(theta);
print('Expected theta (approx):');
print(' -25.161\n 0.206\n 0.201');



## ============== Part 4: Predict and Accuracies ==============
prob = sigmoid(np.dot(np.array([1, 45, 85]), theta))
print("For a student with scores 45 and 85, we predict an admission probability of", round(prob, 3))
print('Expected value: 0.775 +/- 0.002')

# Compute accuracy on our training set
p = predict(theta, X);

print('Train Accuracy:', sum(p == y) * 100 / X.shape[0]);
print('Expected accuracy (approx): 89.0');
















