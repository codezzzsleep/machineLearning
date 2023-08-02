import matplotlib.pyplot as plt
import numpy as np

# 定义三角形的三个顶点
vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])

# 计算三角形的重心
centroid = vertices.mean(axis=0)

# 将坐标系的原点移动到重心
vertices -= centroid

# 画出以重心为原点的三角形
plt.figure()
plt.fill(vertices[:, 0], vertices[:, 1], edgecolor='r', fill=False)

# 画出三个顶点和重心的连线
for v in vertices:
    plt.plot([0, v[0]], [0, v[1]], 'k--')

# 设置坐标轴的范围和标签
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

# 显示图形
plt.show()