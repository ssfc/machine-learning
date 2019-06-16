import matplotlib.pyplot as plt  # 绘图库
import numpy as np
import scipy.optimize as op

# import user defined functions;


def plotData(X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    #print(pos)

    plt.scatter(X[pos, 0], X[pos, 1], color='b', alpha=0.6, label='y=1')  # 绘制散点图，透明度为0.6
    plt.scatter(X[neg, 0], X[neg, 1], color='y', alpha=0.6, label='y=0')  # 绘制散点图，透明度为0.6

    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

    plt.legend(loc='upper right')
    
    plt.show()



def sigmoid(z):
    return 1/(1 + np.exp(-z))







data = np.loadtxt('ex2data2.txt')
X = data[:, [0, 1]]

y = data[:, 2]
y = y.reshape((X.shape[0],1))

plotData(X, y)


















