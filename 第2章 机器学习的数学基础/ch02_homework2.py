# 第二章
# 习题二代码

# 选项A 两个对角矩阵之间相乘一定可交换
# 代码验证：
import numpy as np

D1 = np.diag([1, 2, 3])
D2 = np.diag([4, 5, 6])

result1 = np.dot(D1, D2)
result2 = np.dot(D2, D1)

# 输出结果
print("D1 * D2:\n", result1)
print("D2 * D1:\n", result2)

# 检查两个乘积是否相等
print("Are D1 * D2 and D2 * D1 equal? ", np.array_equal(result1, result2))

# 选项B 矩阵与向量的乘法满足分配律
import numpy as np
# 定义一个矩阵A
A = np.array([[1, 2],[3, 4]])
# 定义两个向量x, y
x = np.array([2, 2])
y = np.array([3, 3])

# 分别计算分配律两边
result3 = A.dot(x+y)
result4 = A.dot(x) + A.dot(y)

print("分配律左边:\n", result3)
print("分配律右边:\n", result4)

# 检查两边是否相等
print("Are they equal? ", np.array_equal(result3, result4))

# 选项C 矩阵对向量的点乘满足结合律 错误 因为点积内积可能是标量
import numpy as np

A = np.array([[1, 2], [3, 4]])

x = np.array([1, 0])
y = np.array([0, 1])

product1 = (A @ x).dot(y)
product2 = A * x.dot(y)

print("x.dot(y):", x.dot(y))
print("Product 1:", product1)
print("Product 2:", product2)

# 检查两个乘积是否相等
print("Are they equal? ", np.array_equal(product1, product2))