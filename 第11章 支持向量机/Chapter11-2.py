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

def SMO(x, y, ker, C, max_iter):
    '''
    SMO算法
    x，y：样本的值和类别
    ker：核函数，与线性回归中核函数的含义相同
    C：惩罚系数
    max_iter：最大迭代次数
    '''
    # 初始化参数
    m = x.shape[0]
    alpha = np.zeros(m)
        
    # 预先计算所有向量的两两内积，减少重复计算
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i, j] = ker(x[i], x[j])
    
    for l in trange(max_iter):
        # 开始迭代
        for i in range(m):
            # 有m个参数，每一轮迭代中依次更新
            # 固定参数alpha_i与另一个随机参数alpha_j，并且保证i与j不相等
            j = np.random.choice([l for l in range(m) if l != i])
            
            # 用-b/2a更新alpha_i的值
            eta = K[j, j] + K[i, i] - 2 * K[i, j] # 分母
            e_i = np.sum(y * alpha * K[:, i]) - y[i] # 分子
            e_j = np.sum(y * alpha * K[:, j]) - y[j]
            alpha_i = alpha[i] + y[i] * (e_j - e_i) / (eta + 1e-5) # 防止除以0
            zeta = alpha[i] * y[i] + alpha[j] * y[j]

            # 将alpha_i和对应的alpha_j保持在[0,C]区间
            # 0 <= (zeta - y_j * alpha_j) / y_i <= C
            if y[i] == y[j]:
                lower = max(0, zeta / y[i] - C)
                upper = min(C, zeta / y[i])
            else:
                lower = max(0, zeta / y[i])
                upper = min(C, zeta / y[i] + C)
            alpha_i = np.clip(alpha_i, lower, upper)
            alpha_j = (zeta - y[i] * alpha_i) / y[j]
            
            # 更新参数
            alpha[i], alpha[j] = alpha_i, alpha_j

    return alpha

# 然后，依次用这4种核心函数计算SVM在双螺旋数据集上的分类结果。
kernels = [
    simple_poly_kernel(3), 
    rbf_kernel(0.1), 
    cos_kernel, 
    sigmoid_kernel(1, -1)
]

ker_names = ['Poly(3)', 'RBF(0.1)', 'Cos', 'Sigmoid(1,-1)']
C = 1e8
max_iter = 500

# 绘图准备，构造网格
plt.figure()
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs = axs.flatten()
cmap = ListedColormap(['coral', 'royalblue'])

# 开始求解 SVM
for i in range(len(kernels)):
    print('核函数：', ker_names[i])
    alpha = SMO(x, y, kernels[i], C=C, max_iter=max_iter)
    sup_idx = alpha > 1e-5 # 支持向量的系数不为零
    sup_x = x[sup_idx] # 支持向量
    sup_y = y[sup_idx]
    sup_alpha = alpha[sup_idx]
    
    # 用支持向量计算 w^T*x
    def wx(x_new):
        s = 0
        for xi, yi, ai in zip(sup_x, sup_y, sup_alpha):
            s += yi * ai * kernels[i](xi, x_new)
        return s
    
    # 计算b*
    neg = [wx(xi) for xi in sup_x[sup_y == -1]]
    pos = [wx(xi) for xi in sup_x[sup_y == 1]]
    b = -0.5 * (np.max(neg) + np.min(pos))
    
    