# 假设在线性回归问题中，数据集有两个样本${x_1=(1,1,1),y_1=0}$和${x_2=(0,1,2),y_2=1}$，
# 尝试用解析方式计算线性回归的参数$\theta$。计算中是否遇到了问题？  

# &emsp;&emsp;$X^{T}X$所得的矩阵是奇异矩阵，不可逆，无法直接适用解析方式求解。

import numpy as np

X = np.array([[1, 1, 1], [0, 1, 2]])

x_T_x = X.T @ X

print(np.linalg.det(x_T_x)) # 行列式为0不可逆

# ——————————————————————————————————
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x_train = np.array([[1, 1, 1], [0, 1, 2]])
y_train = np.array([[0],[1]])

linreg = LinearRegression()
linreg.fit(x_train, y_train)

print('回归系数：', linreg.coef_, linreg.intercept_)
y_pred = linreg.predict(x_train)

rmse_loss = np.sqrt(np.square(y_pred-y_train).mean())
print('RMSE:', rmse_loss)