from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np


# 从源文件加载数据，并输出查看数据的各项特征
lines = np.loadtxt('./Chapter05/USA_Housing.csv', delimiter=',', dtype='str')
header = lines[0]
lines = lines[1:].astype(float)
print('数据特征：', ', '.join(header[:-1]))
print('数据标签：', header[-1])
print('数据总条数：', len(lines))


# 数据归一化
scaler = StandardScaler()
scaler.fit(lines) # 使用所有数据计算均值和方差
lines_scaled = scaler.transform(lines)

# 划分输入和标签
x, y = lines_scaled[:, :-1], lines_scaled[:, -1]

# 定义线性回归模型
model = LinearRegression()

# 使用交叉验证来选择最佳模型
cv_scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')
mse_scores = -cv_scores  # 取负值得到均方误差

print("各模型的均方误差：", mse_scores)

best_mse = mse_scores.max()  # 选择均方误差最小的模型
best_model_index = mse_scores.argmax()  # 最佳模型对应的索引

print("最佳模型的均方误差：", best_mse)

# 在测试集上评估最佳模型
train_size = len(lines) * 4 // 5  # 使用80%的数据作为训练集
test_size = len(lines) - train_size  # 剩下的数据作为测试集
train, test = lines_scaled[:train_size], lines_scaled[train_size:]
x_train, y_train = train[:, :-1], train[:, -1]
x_test, y_test = test[:, :-1], test[:, -1]

best_model = model.fit(x_train, y_train)  # 在训练集上训练最佳模型
y_pred = best_model.predict(x_test)  # 在测试集上预测
mse_test = mean_squared_error(y_test, y_pred)  # 计算在测试集上的均方误差
print("最佳模型在测试集上的均方误差：", mse_test)

# 输出最佳模型的参数
print("最佳模型的参数：")
print("系数（斜率）：", best_model.coef_)
print("截距：", best_model.intercept_)
