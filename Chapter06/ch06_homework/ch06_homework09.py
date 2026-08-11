# 实现多分类逻辑回归
import numpy as np
import matplotlib.pyplot as plt

# 1. 数据加载与预处理
data = np.loadtxt("/workspace/Quant/Hands-on-ML/Chapter06/ch06_homework/lr_dataset.csv", delimiter=',')
X = data[:, :2]  # 特征：x, y坐标
y = data[:, 2].astype(int)  # 标签
K = len(np.unique(y))  # 类别数
y_onehot = np.eye(K)[y]  # 独热编码
N = X.shape[0]  # 样本数量
# 独热编码 (One-Hot Encoding)：将离散的标签转换为向量形式。例如，如果有3个类（0,1,2），标签 1 会被转换为 [0, 1, 0]。这是为了配合 Softmax 输出概率分布进行交叉熵计算。


# 2. 参数初始化
D = X.shape[1]  # 特征维度
W = np.random.randn(D, K-1)  # K-1个参数向量（问题8结论）
b = np.zeros(K-1)  # 偏置项

# 3. 超参数设置
learning_rate = 0.1
max_iter = 1000
loss_history = []

# 4. 训练循环
for epoch in range(max_iter):
    # 前向传播
    z = X.dot(W) + b  # (N, K-1)
    z = np.hstack([z, np.zeros((X.shape[0], 1))])  # 添加参考类别z=0
    z_max = np.max(z, axis=1, keepdims=True)  # 数值稳定性
    exp_z = np.exp(z - z_max)
    probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # 计算损失
    loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-8), axis=1))
    loss_history.append(loss)

    # 反向传播
    dz = (probs - y_onehot) / N  # 梯度对z的导数
    dW = X.T.dot(dz[:, :K-1])  # 前K-1类的梯度
    db = np.sum(dz[:, :K-1], axis=0)

        # 参数更新
    W -= learning_rate * dW
    b -= learning_rate * db
    
    # 打印训练过程
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss:.4f}')

# 5. 评估模型
z_pred = X.dot(W) + b
z_pred = np.hstack([z_pred, np.zeros((X.shape[0], 1))])
probs_pred = np.exp(z_pred - np.max(z_pred, axis=1, keepdims=True))
probs_pred /= np.sum(probs_pred, axis=1, keepdims=True)
y_pred = np.argmax(probs_pred, axis=1)
accuracy = np.mean(y_pred == y)
print(f'\n训练准确率: {accuracy:.4f}')

# 6. 可视化决策边界
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_grid = np.c_[xx.ravel(), yy.ravel()]

# 预测网格点
z_grid = X_grid.dot(W) + b
z_grid = np.hstack([z_grid, np.zeros((X_grid.shape[0], 1))])
probs_grid = np.exp(z_grid - np.max(z_grid, axis=1, keepdims=True))
probs_grid /= np.sum(probs_grid, axis=1, keepdims=True)
y_grid = np.argmax(probs_grid, axis=1).reshape(xx.shape)

# 绘制结果
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, y_grid, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Multi-class decision boundary')
# plt.show()
plt.savefig("/workspace/Quant/Hands-on-ML/Chapter06/ch06_homework/Multi_class_decision_boundary.png")


