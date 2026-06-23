# 证明 原矩阵的转置矩阵的逆矩阵 = 原矩阵的逆矩阵的转置矩阵

# 代码验证：
import numpy as np

# 生成一个可逆的随机矩阵
A = np.random.rand(3, 3)
while np.linalg.det(A) == 0: # 确保矩阵行列式非0，矩阵可逆
    A = np.random.rand(3, 3)

# 计算转置和逆
A_T = A.T
inv_A = np.linalg.inv(A)
inv_A_T = inv_A.T  # (A^{-1})^T
inv_A_T_2 = np.linalg.inv(A_T)

# 验证两者是否相等
print("原矩阵的转置矩阵的逆矩阵  原矩阵的逆矩阵的转置矩阵 是否相等:", np.allclose(inv_A_T, inv_A_T_2))