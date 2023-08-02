import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats

# 假设我们有一组服从 t 分布的数据
np.random.seed(0)
data = np.random.standard_t(df=10, size=100)

# 创建 QQ 图
sm.qqplot(data, stats.t, distargs=(10,), line='45')  # distargs 参数用来指定 t 分布的自由度
plt.show()