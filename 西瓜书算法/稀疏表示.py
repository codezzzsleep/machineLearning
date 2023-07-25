import numpy as np


def sparse_representation(X, k):
    m, n = X.shape
    sparse_X = np.zeros((m, n))  # 创建与输入特征矩阵相同大小的全零矩阵

    for i in range(m):  # 针对每个样本
        abs_values = np.abs(X[i])  # 计算特征向量的绝对值
        indices = np.argsort(abs_values)[::-1][:k]  # 取绝对值后按降序排列并选择前k个索引

        # 将选定的索引位置更新为原始特征向量中的值
        sparse_X[i][indices] = X[i][indices]

    return sparse_X


# 示例数据集
X = np.array([[1, 2, 0, 3], [4, 0, 5, 6], [7, 8, 9, 10]])

# 设置稀疏度参数
k = 2

# 执行稀疏表示
sparse_X = sparse_representation(X, k)
print("原始表示结果：")
print(X)
print("稀疏表示结果：")
print(sparse_X)
