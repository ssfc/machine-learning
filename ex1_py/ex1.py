import numpy as np
import matplotlib.pyplot as plt  # 绘图库

import computeCost


data = np.loadtxt('ex1data1.txt')
m = data.shape[0]
y = data[:, 1]


plt.scatter(data[:, 0], data[:, 1], alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
#plt.show()


X = np.c_[np.ones([m, 1]), data[:, 0]]    # Add a column of ones to x
theta = np.zeros([2, 1]);    # initialize fitting parameters
#theta[0] = 1;
#theta[1] = 2;



J = computeCost.computeCost(X, y, theta);
print(J)
