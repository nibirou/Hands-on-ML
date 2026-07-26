# 自主实现一个线性回归，并进行多维度评估
import numpy as np
import matplotlib.pyplot as plt

# 数据集
X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
Y = np.array([0.3, 0.35, 0.41, 0.48, 0.54])

# 参数
theta0 = 0.0
theta1 = 0.0

# 线性回归模型
def linear_regression(x, t0, t1):
    return t0 + t1 * x

# 损失函数MSE
def Loss_function(theta, X, y):
    t0, t1 = theta
    predictions = linear_regression(X, t0, t1)
    return np.sum((predictions - y) ** 2) / (2 * len(X))

# 梯度下降函数
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(X)
    for i in range(iterations):
        predictions = linear_regression(X, theta[0], theta[1])
        grad_t0 = np.sum(predictions - y) / m
        grad_t1 = np.sum((predictions - y) * X) / m
        theta[0] -= learning_rate * grad_t0
        theta[1] -= learning_rate * grad_t1
    return theta

# 学习率和迭代次数
learning_rate = 0.15
iterations = 1000

optimized_theta = gradient_descent(X, Y, [theta0, theta1], learning_rate, iterations)

print(f"theta0: {optimized_theta[0]}")
print(f"theta1: {optimized_theta[1]}")

# 使用优化后的参数进行预测
predictions = linear_regression(X, optimized_theta[0], optimized_theta[1])
print("Predicted values:")
print(predictions)

# 设置几个测试点
X_test = np.array([0.6, 0.7, 0.05])
# 使用线性回归模型预测测试点的值
predictions_test = optimized_theta[0] + optimized_theta[1] * X_test


# 可视化真实数据点和拟合直线以及测试点
# 可视化真实数据点、拟合直线和测试点
plt.figure(figsize=(8, 6))
plt.scatter(X, Y, color='blue', label='Data Points')
plt.scatter(X_test, predictions_test, color='green', label='Test Predictions')
plt.plot(X_test, optimized_theta[0] + optimized_theta[1] * X_test, color='lightgray', label='Test Fitted Line')
plt.plot(X, optimized_theta[0] + optimized_theta[1] * X, color='red', label='Fitted Line')
plt.title('Linear Regression Fit')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
# plt.show()
plt.savefig('/workspace/Quant/Hands-on-ML/第4章 线性回归/ch04_homework/Linear_Regression_Fit.png')

# 最小二乘法计算进行验证
# 增加一列全为1的偏置列
X_b = np.c_[np.ones((X.shape[0], 1)), X.reshape(-1, 1)]

# 使用最小二乘法求解参数
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)
# 打印参数
print("参数：", theta_best)

# 预测新数据
x_new = 0.6
X_new_b = np.array([[1, x_new]])
y_predict = X_new_b.dot(theta_best)
print("预测值：", y_predict[0])

# &emsp;&emsp;我们画图看看MSE和参数$\theta0,\theta1$的关系。  
# &emsp;&emsp;可以看出随着参数拟合到合理的范围，MSE的值比较小，在不合理的范围时，MSE的值会比较大。
# 数据集
X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
Y = np.array([0.3, 0.35, 0.41, 0.48, 0.54])

# 参数范围
theta0_range = np.linspace(-2, 2, 100)
theta1_range = np.linspace(-2, 2, 100)

# 计算损失函数的值
loss_values = np.zeros((len(theta0_range), len(theta1_range)))
for i, theta0 in enumerate(theta0_range):
    for j, theta1 in enumerate(theta1_range):
        loss_values[i, j] = np.sum((theta0 + theta1 * X - Y) ** 2) / (2 * len(X))

# 绘制3D图
theta0_grid, theta1_grid = np.meshgrid(theta0_range, theta1_range)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0_grid, theta1_grid, loss_values.T, cmap='viridis', alpha=0.8)
ax.set_xlabel('Theta0')
ax.set_ylabel('Theta1')
ax.set_zlabel('Loss')
ax.set_title('Loss Function Landscape')
plt.savefig('/workspace/Quant/Hands-on-ML/第4章 线性回归/ch04_homework/Loss_Function_Landscape.png')
# plt.show()

# 绘制填充轮廓图
plt.figure(figsize=(10, 6))
contour = plt.contourf(theta0_grid, theta1_grid, loss_values, levels=20, cmap='viridis')
plt.colorbar(contour)
plt.xlabel('Theta0')
plt.ylabel('Theta1')
plt.title('Loss Function Contour')
plt.savefig('/workspace/Quant/Hands-on-ML/第4章 线性回归/ch04_homework/Loss_Function_Contour.png')
# plt.show()

# &emsp;&emsp;下面我们画图看看不同起始参数位置和学习率对参数学习轨迹的影响。  
# &emsp;&emsp;可以看到不同的初始参数位置、学习率会导致求得的“最优解”在不同的位置，同时迭代次数和速度也会有影响。
# 学习率较大会更快收敛到最优解同时轨迹更加陡峭，学习率较小会收敛到局部最优并且轨迹比较平缓。
X = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
Y = np.array([0.3, 0.35, 0.41, 0.48, 0.54])

# 损失函数MSE
def loss_function(theta, X, y):
    predictions = theta[0] + theta[1] * X
    return np.sum((predictions - y) ** 2) / (2 * len(X))

# 梯度下降函数
def gradient_descent(X, y, theta, learning_rate, iterations):
    