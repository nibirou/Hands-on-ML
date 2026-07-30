# 构建一个多元线性回归，并做ch04_homework4类似求解

import numpy as np
import matplotlib.pyplot as plt

N = 3
K = 4 # N元，K个样本
X = np.random.rand(N, K)
Y = np.random.rand(N, K)

# 参数
theta = np.zeros([N, 1])
t0 = np.zeros([1, 1])

# 线性回归模型
def linear_regression(x, t0, theta):
    return t0 + theta * x

# 损失函数MSE
def Loss_function(t0, theta, X, y):
    predictions = linear_regression(X, t0, theta)
    return np.sum((predictions - y) ** 2) / (2 * K)

# 梯度下降函数
def gradient_descent(X, y, t0, theta, learning_rate, iterations):
    for i in range(iterations):
        predictions = linear_regression(X, t0, theta)
        grad_t0 = np.sum(predictions - y) / K
        grad_t = np.sum((predictions - y) * X) / K
        t0 -= learning_rate * grad_t0
        theta -= learning_rate * grad_t
    return t0, theta

# 学习率和迭代次数
learning_rate = 0.15
iterations = 1000

t0, theta = gradient_descent(X, Y, t0, theta, learning_rate, iterations)

print(f"t0: {t0}")
print(f"theta: {theta}")

# 使用优化后的参数进行预测
predictions = linear_regression(X, t0, theta)
print("Predicted values:")
print(predictions)


# 上面这个代码存在几个问题
# 1. 样本与特征的维度混淆
# 在机器学习中，通常行代表样本，列代表特征。
# 你的注释写着“N元（特征数），K个样本”，那么 X 的形状应该是 (K, N)，而不是 (N, K)。
# 目标值 Y 应该是每个样本对应一个值，形状应为 (K, 1)，而你写成了 (N, K)，这变成了多变量回归。

# 误用逐元素乘法 * 代替矩阵乘法 @
# 在 NumPy 中，* 是逐元素相乘（对应位置相乘），而多元线性回归需要的是矩阵乘法（点积）。
# 你的模型 t0 + theta * x 是错误的。正确的矩阵表达应该是 t0 + X @ theta。

# 梯度计算维度不匹配
# 你的 grad_t = np.sum((predictions - y) * X) / K 中，因为用了 *，最后 np.sum 会把整个矩阵压缩成一个标量
# 但你的 theta 是 (N, 1) 的向量，用一个标量去更新向量，数学上完全错误。正确的梯度公式是
# 代码应写为 (X.T @ error) / K

# 对纯随机数做回归没有意义
# 你的 X 和 Y 都是 np.random.rand 生成的纯随机数，它们之间没有线性关系。即使代码写对了，求出来的 theta 也没有实际意义。
# 通常需要自己构造一个真实的 theta 来生成 Y（加上一点噪声），这样回归才有验证价值。


# 修正了维度、矩阵乘法，并加入了损失函数可视化
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. 数据准备 =================
K = 100  # 样本数量 (通常样本数远大于特征数)
N = 3    # 特征数量 (N元)

# X 的形状应为 (K, N)，Y 的形状应为 (K, 1)
X = np.random.rand(K, N)
# 构造真实的参数来生成 Y，让回归有意义 (加入少量高斯噪声)
true_theta = np.array([[2.0], [3.0], [-1.5]])
true_t0 = 1.5
Y = true_t0 + X @ true_theta + np.random.randn(K, 1) * 0.1 

# ================= 2. 参数初始化 =================
theta = np.zeros([N, 1])
t0 = np.zeros([1, 1])

# ================= 3. 模型与损失函数 =================
# 线性回归模型 (注意使用矩阵乘法 @)
def linear_regression(X, t0, theta):
    return t0 + X @ theta

# 损失函数 MSE
def Loss_function(t0, theta, X, y):
    predictions = linear_regression(X, t0, theta)
    return np.sum((predictions - y) ** 2) / (2 * K)

# ================= 4. 梯度下降 =================
def gradient_descent(X, y, t0, theta, learning_rate, iterations):
    K = X.shape[0]
    loss_history = []
    for i in range(iterations):
        predictions = linear_regression(X, t0, theta)
        error = predictions - y  # 形状: (K, 1)

        # 计算梯度 (注意使用矩阵乘法 @ 和转置 .T)
        grad_t0 = np.sum(error) / K # 标量
        grad_t = (X.T @ error) / K # X.T 是 (N, K), error 是 (K, 1) -> 结果 (N, 1)

        # 更新参数
        t0 -= learning_rate * grad_t0
        theta -= learning_rate * grad_t

        # 记录损失以便画图
        if i % 10 == 0:
            loss_history.append(Loss_function(t0, theta, X, y))
    return t0, theta, loss_history

# ================= 5. 训练模型 =================
learning_rate = 0.5  # 学习率
iterations = 1000    # 迭代次数

t0_opt, theta_opt, loss_history = gradient_descent(X, Y, t0, theta, learning_rate, iterations)

# ================= 6. 结果输出 =================
print("========== 优化结果 ==========")
# print(f"真实 t0: {true_t0[0,0]:.4f}  |  预测 t0: {t0_opt[0,0]:.4f}")
print(f"真实 theta: {true_theta.flatten()}")
print(f"预测 theta: {theta_opt.flatten()}")

# ================= 7. 可视化 =================
plt.figure(figsize=(10, 4))

# 图1：损失下降曲线
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='b', linewidth=2)
plt.title("MSE Loss Curve", fontsize=12)
plt.xlabel("Iterations", fontsize=10)
plt.ylabel("Loss", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# 图2：预测值 vs 真实值
plt.subplot(1, 2, 2)
predictions = linear_regression(X, t0_opt, theta_opt)
plt.scatter(Y, predictions, alpha=0.6, color='green')
# 画一条 y=x 的参考线
min_val = min(Y.min(), predictions.min())
max_val = max(Y.max(), predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
plt.title("Predicted vs True Values", fontsize=12)
plt.xlabel("True Y", fontsize=10)
plt.ylabel("Predicted Y", fontsize=10)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
# plt.show()
plt.savefig("/workspace/Quant/Hands-on-ML/第4章 线性回归/ch04_homework/homework5_Linear_Regression_Fit.png")