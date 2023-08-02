import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("penguins.csv")
df = df.drop(df.columns[0], axis=1)
x = df[['bill_length_mm']].values
y = df[['bill_depth_mm']].values
z = df[['flipper_length_mm']].values
# print(x)
# print(y)
# print(z)


# 创建一个新的图形
fig = plt.figure()

# 创建一个3D子图
ax = fig.add_subplot(111, projection='3d')

# 在3D子图上创建一个散点图
ax.scatter(x, y, z)

# 设置坐标轴的标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()

# barycentric_coordinates = np.vstack((x, y, z))
# barycentric_coordinates = barycentric_coordinates / np.sum(barycentric_coordinates, axis=0)
#
# # 创建一个三角形
# triangle = tri.Triangulation(barycentric_coordinates[0, :], barycentric_coordinates[1, :])
#
# # 绘制三角形
# plt.triplot(triangle)
#
# # 在三角形上绘制数据点
# plt.scatter(barycentric_coordinates[0, :], barycentric_coordinates[1, :], c=z)
#
# # 显示图形
# plt.show()
totals = x + y + z

# 将每个坐标值除以总值，使得新的x，y，z的和为1
x /= totals
y /= totals
z /= totals

# 用新的x，y值作为三角形中的坐标，z值可以用于表示颜色或大小等其他属性


# 创建一个散点图
plt.scatter(x, y, c=z)

# 设置坐标轴的范围
plt.xlim(0.12, 0.25)
plt.ylim(0.04, 0.10)

# 显示图形
plt.show()
