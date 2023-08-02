import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# 假设你已经有了x，y，z三个Numpy数组，每个数组包含100个元素
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

# 计算barycentric coordinates
barycentric_coordinates = np.vstack((x, y, z))
barycentric_coordinates = barycentric_coordinates / np.sum(barycentric_coordinates, axis=0)

# 创建一个三角形
triangle = tri.Triangulation(barycentric_coordinates[0, :], barycentric_coordinates[1, :])

# 绘制三角形
plt.triplot(triangle)

# 在三角形上绘制数据点
plt.scatter(barycentric_coordinates[0, :], barycentric_coordinates[1, :], c=z)

# 显示图形
plt.show()