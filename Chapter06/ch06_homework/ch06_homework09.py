# 实现多分类逻辑回归
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
data = np.loadtxt("./lr_dataset.csv", delimiter=',')
X = data[:, :2]  # 特征：x, y坐标
y = data[:, 2].astype(int)  # 标签
K = len(np.unique(y))  # 类别数
y_onehot = np.eye(K)[y]  # 独热编码

