import matplotlib.pyplot as plt
import numpy as np

# 定义三维数据点P
P = np.array([0.2, 0.3, 0.5])

# 检查点P是否满足x + y + z = 1
assert np.isclose(P.sum(), 1), "点P必须满足x + y + z = 1"

# 定义三角形的三个顶点
vertices = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

# 计算三角形的重心
centroid = vertices.mean(axis=0)

# 将坐标系的原点移动到重心
vertices -= centroid

# 画出以重心为原点的三角形
plt.figure()
plt.fill(vertices[:, 0], vertices[:, 1], edgecolor='r', fill=False)

# 将点P投影到三角形坐标系中
P_prime = P[:2] - centroid

# 画出点P
plt.plot(P_prime[0], P_prime[1], 'bo')

# 设置坐标轴的范围和标签
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

# 显示图形
plt.show()