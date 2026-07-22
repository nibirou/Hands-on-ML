# 证明 二元函数函数的梯度方向是函数值上升最快的方向
# 代码验证：以 $f(x, y) = x^2 + y^3$ 为例，在点 $(1, 1)$ 处计算梯度，
# 并比较沿不同方向的方向导数，验证最大值出现在梯度方向。

import numpy as np
import matplotlib.pyplot as plt

# 定义函数及其梯度
def f(x, y):
    return x**2 + y**3

def gradient_f(x, y):
    return np.array([2*x, 3*y**2])

# 测试点
x0, y0 = 1, 1
grad = gradient_f(x0, y0)
grad_dir = grad / np.linalg.norm(grad)  # 梯度方向的单位向量
print(grad_dir)

# 生成不同方向的方向向量（单位向量）
theta = np.linspace(0, 2*np.pi, 1000)
directions = np.column_stack((np.cos(theta), np.sin(theta)))
print(directions)

# 计算方向导数
directional_derivatives = np.sum(grad * directions, axis=1)

# 找到最大值对应的角度
max_idx = np.argmax(directional_derivatives)
max_theta = theta[max_idx]

# 输出结果（用反三角函数arctan，然后弧度转角度）
print(f"梯度方向：θ = {np.degrees(np.arctan(grad_dir[1]/grad_dir[0])):.2f}°")
print(f"方向导数最大值对应的角度：θ = {np.degrees(max_theta):.2f}°")


# 绘制方向导数随角度的变化
plt.figure(figsize=(8, 5))
plt.plot(np.degrees(theta), directional_derivatives, label="function")
plt.axvline(np.degrees(max_theta), color='r', linestyle='--', label="max")
plt.xlabel("θ°")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
plt.savefig('./第2章 机器学习的数学基础/ch02_homework5.png')