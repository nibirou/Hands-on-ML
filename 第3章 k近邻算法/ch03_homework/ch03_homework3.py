# 习题3
# KNN 算法中，我们采用了最常用的欧氏距离作为寻找邻居的标准。
# 在哪些场景下，我们可能会用到其他距离度量，例如曼哈顿距离（Manhattan distance）

# 把第 3 节实验中的距离改为曼哈顿距离，观察对分类效果的影响。

import matplotlib.pyplot as plt
import numpy as np
import os

m_x = np.loadtxt('./第3章 k近邻算法/mnist_x', delimiter=' ')
m_y = np.loadtxt('./第3章 k近邻算法/mnist_y')

# 数据集可视化
data = np.reshape(np.array(m_x[0], dtype=int), [28, 28])
plt.figure()
plt.imshow(data, cmap='gray')

# 将数据集分为训练集和测试集
ratio = 0.8
split = int(len(m_x) * ratio)

# 打乱数据
np.random.seed(0)
idx = np.random.permutation(np.arange(len(m_x)))

m_x = m_x[idx]
m_y = m_y[idx]
x_train, x_test = m_x[:split], m_x[split:]
y_train, y_test = m_y[:split], m_y[split:]

# 曼哈顿距离
def distance(a, b):
    return sum(abs(a - b))

class KNN:
    def __init__(self, k, label_num):
        self.k = k
        self.label_num = label_num # 类别的数量
    
    def fit(self, x_train, y_train):
        # 在类中保存训练数据
        self.x_train = x_train
        self.y_train = y_train
        
    def get_knn_indices(self, x):
        # 获取距离目标样本点最近的K个样本点的标签
        # 计算已知样本的距离
        dis = list(map(lambda a: distance(a, x), self.x_train))
        # 按距离从小到大排序，并得到对应的下标
        knn_indices = np.argsort(dis)
        # 取最近的K个
        knn_indices = knn_indices[:self.k]
        return knn_indices
    
    def get_label(self, x):
        # 对KNN方法的具体实现，观察K个近邻并使用np.argmax获取其中数量最多的类别
        knn_indices = self.get_knn_indices(x)
        label_statistic = np.zeros()