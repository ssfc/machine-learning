import numpy as np
import matplotlib.pyplot as plt  # 绘图库


def plotData(X, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    #print(pos)


    plt.scatter(X[pos, 0], X[pos, 1], color='b', alpha=0.6, label='Admitted')  # 绘制散点图，透明度为0.6
    plt.scatter(X[neg, 0], X[neg, 1], color='y', alpha=0.6, label='Not admitted')  # 绘制散点图，透明度为0.6

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')


    plt.legend(loc='upper right')


    
    plt.show()











