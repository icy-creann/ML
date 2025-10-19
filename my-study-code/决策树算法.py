"""
决策树(Decision Tree)是一种常用的机器学习算法,广泛应用于分类和回归问题。

决策树通过树状结构来表示决策过程,每个内部节点代表一个特征或属性的测试,每个分支代表测试的结果,每个叶节点代表一个类别或值。
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier,plot_tree
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


model = DecisionTreeClassifier()
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


plt.figure(figsize=(20,10))
plot_tree(model, filled=True, feature_names=['Age', 'Estimated Salary'], class_names=['Not Purchased', 'Purchased'])
plt.title('Decision Tree')
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()
