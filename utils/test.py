import numpy as np

# 创建一个示例矩阵
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# 对行进行排列组合
permuted_rows = np.random.permutation(matrix)

# 对列进行排列组合
permuted_columns = np.random.permutation(matrix.T).T

print("原始矩阵：")
print(matrix)
print("行排列组合后的矩阵：")
print(permuted_rows)
print("列排列组合后的矩阵：")
print(permuted_columns)
