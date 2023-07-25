import numpy as np


def lasso_regression(X, y, alpha, num_iters):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(num_iters):
        # 计算预测值
        y_pred = np.dot(X, theta)

        # 更新参数（考虑梯度的符号）
        gradient = 1 / m * np.dot(X.T, y_pred - y)
        theta -= alpha * gradient + alpha * np.sign(theta)

    return theta


# 示例数据集
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array([1, 2, 3, 4])

# 添加偏置项
X = np.c_[np.ones(X.shape[0]), X]

# 设置超参数
alpha = 0.1
num_iters = 500

# 执行Lasso回归
theta = lasso_regression(X, y, alpha, num_iters)

print("特征权重：", theta[1:])
