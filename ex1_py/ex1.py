import numpy as np
import matplotlib.pyplot as plt  # 绘图库

# import user defined functions;
import computeCost
import gradientDescent

## ======================= Part 2: Plotting =======================
data = np.loadtxt('ex1data1.txt')
m = data.shape[0]
y = data[:, 1]


## =================== Part 3: Cost and Gradient descent ===================
X = np.c_[np.ones([m, 1]), data[:, 0]]    # Add a column of ones to x
theta = np.zeros([2, 1]);    # initialize fitting parameters
#theta[0] = 1;
#theta[1] = 2;


# Some gradient descent settings
iterations = 1500;
alpha = 0.01;


print("Testing the cost function ...")
# compute and display initial cost
J = computeCost.computeCost(X, y, theta);
print("With theta = [0 ; 0]\nCost computed = ", J)
print("Expected cost value (approx) 32.07")


# further testing of the cost function
J = computeCost.computeCost(X, y, [[-1], [2]])
print("With theta = [-1 ; 2]\nCost computed = ", J)
print("Expected cost value (approx) 54.24")

# run gradient descent
print("\nRunning Gradient Descent ...")
theta = gradientDescent.gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print("Theta found by gradient descent:")
print(theta)
print("Expected theta values (approx)")
print(" -3.6303\n  1.1664\n\n")

# Plot the linear fit
plt.scatter(X[:, 1], y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
plt.plot(X[:, 1], np.dot(X, theta), color='r', alpha=0.6)  # connect dots with line
plt.show()


# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot([1, 3.5], theta)
print("For population = 35,000, we predict a profit of \n",
    predict1*10000);
predict2 = np.dot([1, 7], theta)
print("For population = 70,000, we predict a profit of \n",
    predict2*10000);

























