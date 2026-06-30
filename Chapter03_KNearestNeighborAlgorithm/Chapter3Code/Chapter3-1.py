# 使用KNN进行手写数字分类，可视化mnist数据

import matplotlib.pyplot as plt
import numpy as np
import os

# 读入mnist数据集
# m_x.shape = [1000,784]
m_x = np.loadtxt('./第3章 k近邻算法/mnist_x', delimiter=' ')
print(m_x.shape)
m_y = np.loadtxt('./第3章 k近邻算法/mnist_y')

# 数据集可视化
data = np.reshape(np.array(m_x[0], dtype=int), [28, 28])
plt.figure()
plt.imshow(data, cmap='gray')
plt.show()