import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 假设我们有一组服从正态分布的数据
np.random.seed(0)  # 为了结果的可重复性
data = np.random.normal(loc=0.0, scale=1.0, size=100)

# 创建 QQ 图
sm.qqplot(data, line='45')  # line='45' 会添加一个 y=x 参考线
plt.show()