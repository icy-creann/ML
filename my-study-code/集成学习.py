"""
在机器学习领域,集成学习(Ensemble Learning)是一种通过结合多个模型的预测结果来提高整体性能的技术。

集成学习的核心思想是通过多个弱学习器的组合,可以构建一个强学习器。

集成学习的主要目标是通过组合多个模型来提高预测的准确性和鲁棒性。

常见的集成学习方法包括:
Bagging:通过自助采样法(Bootstrap Sampling)生成多个训练集,然后分别训练多个模型,最后通过投票或平均的方式得到最终结果。
Boosting:通过迭代的方式训练多个模型,每个模型都试图纠正前一个模型的错误,最终通过加权投票的方式得到结果。
Stacking:通过训练多个不同的模型,然后将这些模型的输出作为新的特征,再训练一个元模型(Meta-Model)来进行最终的预测。
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

dataset = pd.read_csv(f"ML/my-study-code/datasets/Social_Network_Ads.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:, -1].values

le = LabelEncoder()
X[:, 0] = le.fit_transform(X[:,0])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""
Bagging 集成学习
常见的 Bagging 方法包括 BaggingClassifier 和 RandomForestClassifier。此处展示RandomForestClassifier的使用示例。
"""
bagging_clf = RandomForestClassifier(n_estimators=100, random_state=42)
bagging_clf.fit(X_train, y_train)

y_pred_bagging = bagging_clf.predict(X_test)

accuracy_bagging = accuracy_score(y_test, y_pred_bagging)
print(f"Bagging 模型准确率: {accuracy_bagging*100:.2f}%")
print("Bagging 模型混淆矩阵:")
print(confusion_matrix(y_test, y_pred_bagging))
print("Bagging 模型分类报告:")
print(classification_report(y_test, y_pred_bagging))


"""
Boosting 集成学习
常见的 Boosting 方法包括 AdaBoostClassifier 和 GradientBoostingClassifier。此处展示AdaBoostClassifier的使用示例。
"""
boosting_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42,algorithm='SAMME')
boosting_clf.fit(X_train, y_train)

y_pred_boosting = boosting_clf.predict(X_test)

accuracy_boosting = accuracy_score(y_test, y_pred_boosting)
print(f"Boosting 模型准确率: {accuracy_boosting*100:.2f}%")
print("Boosting 模型混淆矩阵:")
print(confusion_matrix(y_test, y_pred_boosting))
print("Boosting 模型分类报告:")
print(classification_report(y_test, y_pred_boosting))


"""
Stacking 集成学习
常见的 Stacking 方法包括 StackingClassifier。此处展示StackingClassifier的使用示例。
"""
estimators = [
    ('dt', DecisionTreeClassifier(max_depth=1)),
    ('svc', SVC(kernel='linear', probability=True))
]

stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_clf.fit(X_train, y_train)

y_pred_stacking = stacking_clf.predict(X_test)

accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
print(f"Stacking 模型准确率: {accuracy_stacking*100:.2f}%")
print("Stacking 模型混淆矩阵:")
print(confusion_matrix(y_test, y_pred_stacking))
print("Stacking 模型分类报告:")
print(classification_report(y_test, y_pred_stacking))
