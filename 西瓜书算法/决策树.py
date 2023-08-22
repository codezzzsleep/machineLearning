from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

# 加载示例数据集（例如，使用鸢尾花数据集）
iris = datasets.load_iris()
X = iris.data  # 特征
y = iris.target  # 目标变量
# 将数据集拆分为训练集和测试集（70%的数据用于训练，30%用于测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 创建决策树分类器对象
clf = DecisionTreeClassifier()

# 使用训练数据对分类器进行训练
clf.fit(X_train, y_train)
# 使用测试数据进行预测
y_pred = clf.predict(X_test)
# 评估模型的准确率
accuracy = metrics.accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
print(iris)
print(X)
print(y)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
