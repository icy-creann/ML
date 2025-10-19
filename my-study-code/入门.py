"""
ML入门代码示例：使用KNN算法进行分类
共分为6个步骤：
1. 导入库
2. 加载数据
3. 划分数据集
4. 数据标准化和热编码
5. 训练模型
6. 模型预测
"""

#step 1 导入库
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  #划分数据集
from sklearn.preprocessing import StandardScaler  #数据标准化
from sklearn.neighbors import KNeighborsClassifier  #KNN分类器
from sklearn.metrics import accuracy_score, confusion_matrix  #准确率, 混淆矩阵



#step 2 加载数据
dataset = pd.read_csv('ML\my-study-code\datasets\Social_Network_Ads.csv')
X = pd.DataFrame(dataset, columns=dataset.columns[2:-1]) #特征数据,参数1为数据，参数2为数据的列名，该代码等价于X = dataset.iloc[:, :-1]
Y = pd.Series(dataset.iloc[:, -1]) #标签数据
print("特征数据前5行:\n", X.head())
print("标签数据前5行:\n", Y.head())


#step 3 划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


#step 4 数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#step 5 训练模型
knn = KNeighborsClassifier(n_neighbors=3) #实例化KNN分类器，
knn.fit(X_train, Y_train) #训练模型

#step 6 模型预测
Y_pred = knn.predict(X_test) #预测结果
cm = confusion_matrix(Y_test, Y_pred) #计算混淆矩阵
print("混淆矩阵:\n", cm)
accuracy = accuracy_score(Y_test, Y_pred) #计算准确率
print("模型准确率: {:.2f}%".format(accuracy * 100))

