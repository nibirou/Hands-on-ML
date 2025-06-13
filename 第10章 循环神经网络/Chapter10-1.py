# 对于具有序列特征的数据，如温度、文本等，他们具有明显的前后关联。
# 同时这些关联的数据在序列中出现的位置可能间隔非常远，比如文章在开头和结尾描写了同一个事物，如果用CNN来提取关联的话，
# 其卷积核的大小需要和序列的长度匹配。当数据序列较长时，会大大增加网络复杂度和训练难度
# 因此，引入循环神经网络RNN，充分利用数据的序列性质，从前到后分析数据、提取关联

# RNN因为在反向求导时，随着反向传播步数增加，由于求导链式法则，梯度中出现多层激活函数导数与权重项连乘
# 可能会出现梯度消失与梯度爆炸（实际上神经网络都这样）
# 梯度消失，模型收敛速度会变慢；梯度爆炸，模型梯度会迅速发散，参数变化幅度大，不收敛

# 为了防止梯度消失或爆炸，最简单的方法是裁剪梯度，为梯度设置上下限。当梯度过大或者过小时，采用上下限来替代梯度的值
# 还可以选用合适的激活函数并调整网络权重参数的初始值，使得两者乘积稳定在1附近。

# 因此，可以将网络中关联起相邻两步的激活函数和网络权重参数扩展成一个小的网络

# 门控循环单元(gated recurrent unit, GRU)

# 动手实现一个GRU模型，完成简单的时间序列预测任务

# 根据一段连续时间内采集的数据，分析其变化规律，预测数据走向

# <------------------------------------------------------------------------------>
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

# 导入数据集
data = np.loadtxt('./第10章 循环神经网络/sindata_1000.csv', delimiter=',')
num_data = len(data)
split = int(0.8 * num_data)
print(f'数据集大小：{num_data}')
# 数据集可视化
plt.figure()
plt.scatter(np.arange(split), data[:split], color='blue', 
    s=10, label='training set')
plt.scatter(np.arange(split, num_data), data[split:], color='none', 
    edgecolor='orange', s=10, label='test set')
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.legend()
plt.show()
# 分割数据集
train_data = np.array(data[:split])
test_data = np.array(data[split:])

# <------------------------------------------------------------------------------>
# 在训练RNN模型时，虽然我们卡一把每个时间步数t单独输入，得到模型的预测值，但这样无法体现数据的序列相关性质
# 因此，通常会把一段