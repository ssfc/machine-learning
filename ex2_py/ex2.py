import numpy as np
import matplotlib.pyplot as plt  # 绘图库

# import user defined functions;
import plotData



data = np.loadtxt('ex2data1.txt')
X = data[:, [0, 1]]
y = data[:, 2]

## ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")

plotData.plotData(X, y)

## ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in 
#  costFunction.m


































