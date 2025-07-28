# 核函数与松弛变量

# 对于略微有些对于略微有些线性不可分的数据，我们采用松弛变量的方法，仍然可以导出SVM的分隔超平面。然而，当数据的分布更加偏离线性时，可能完全无法用线性的超平面进行有效分类，松弛变量也就失效了。为了更清晰地展示非线性的情况，
# 我们读入双螺旋数据集spiral.csv并绘制数据分布。该数据集包含了在平面上呈螺旋分布的两组点，同类的点处在同一条旋臂上。
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm, trange

data = np.loadtxt('./第11章 支持向量机/spiral.csv', delimiter=',')
print('数据集大小：', len(data))
x = data[:, :2]
y = data[:, 2]

# 数据集可视化
plt.figure()
plt.scatter(x[y == -1, 0], x[y == -1, 1], color='red', label='y=-1')
plt.scatter(x[y == 1, 0], x[y == 1, 1], marker='x', color='blue', label='y=1')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend()
plt.axis('square')
plt.savefig('./第11章 支持向量机/spiral.png')
plt.show()

# 显然，平面上任意一条直线都无法为上图的双螺旋数据集给出分类。因此，我们需要引入非线性的特征映射。
# 在神经网络一章中我们已经提到，非线性函数可以使数据升维，将在低维中线性不可分的数据映射到高维空间，使其变得线性可分。

# SVM在计算时只需要用到支持向量的优点就很明显了。
# 在数据集上建立好SVM模型后，我们可以只保留支持向量，而将剩余的数据都舍弃，减小存储模型的空间占用。

# 现在，我们的任务就变成寻找合适的核函数，来让原本线性不可分的数据线性可分。通常来说，核函数应当衡量向量之间的相似度。

# 简单多项式核
def simple_poly_kernel(d):
    def k(x, y): 
        return np.inner(x, y) ** d
    return k

# RBF核
def rbf_kernel(sigma):
    def k(x, y):
        return np.exp(-np.inner(x - y, x - y) / (2.0 * sigma ** 2))
    return k

# 余弦相似度核
def cos_kernel(x, y):
    return np.inner(x, y) / np.linalg.norm(x, 2) / np.linalg.norm(y, 2)

# sigmoid核
def sigmoid_kernel(beta, c):
    def k(x, y):
        return np.tanh(beta * np.inner(x, y) + c)
    return k

