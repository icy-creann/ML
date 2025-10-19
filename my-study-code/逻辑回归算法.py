"""
逻辑回归(Logistic Regression) 是一种广泛应用于分类问题的统计学习方法，尽管名字中带有"回归"，但它实际上是一种用于二分类或多分类问题的算法。
逻辑回归的目标是找到一个最佳的决策边界，将数据点分为不同的类别。
逻辑回归通过使用逻辑函数（也称为 Sigmoid 函数）将线性回归的输出映射到 0 和 1 之间，从而预测某个事件发生的概率。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv(f'ML\my-study-code\datasets\Social_Network_Ads.csv')
X = dataset.iloc[:, 2:4].values  # 特征变量
Y = dataset.iloc[:, 4].values    # 目标变量

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)


# 可视化决策边界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
                     np.arange(y_min, y_max, 1000))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])# 预测网格点的类别,np.c_用于将两个数组按列连接为一个二维数组
Z = Z.reshape(xx.shape)# 将预测结果转换为与网格点相同的形状

plt.contourf(xx, yy, Z, alpha=0.2)
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', marker='o')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.title('Logistic Regression Decision Boundary')
plt.show()