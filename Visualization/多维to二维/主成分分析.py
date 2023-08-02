import numpy as np
import matplotlib.pyplot as plt
# 假设你已经有了x，y，z三个Numpy数组，每个数组包含100个元素
x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

# 计算每个点的总值
totals = x + y + z

# 将每个坐标值除以总值，使得新的x，y，z的和为1
x /= totals
y /= totals
z /= totals

# 用新的x，y值作为三角形中的坐标，z值可以用于表示颜色或大小等其他属性


# 创建一个散点图
plt.scatter(x, y, c=z)

# 设置坐标轴的范围
plt.xlim(0, 1)
plt.ylim(0, 1)

# 显示图形
plt.show()