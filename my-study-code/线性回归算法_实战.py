import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  #划分数据集
from sklearn.preprocessing import StandardScaler  #数据标准化
from sklearn.linear_model import LinearRegression  #线性回归模型

#加载数据
dataset = {
    '面积':[2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494],
    '价格':[399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999, 212000, 242500]
}

data = pd.DataFrame(dataset)
X = data[['面积']]  #特征数据
Y = data['价格']  #标签数据
# print("特征数据:\n", X)
# print("标签数据:\n", Y)           

#划分数据集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
# print("特征训练数据:\n", X_train)
# print("特征测试数据:\n", X_test)
# print("标签训练数据:\n", Y_train)
# print("标签测试数据:\n", Y_test)

#数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#训练模型
model = LinearRegression() #实例化线性回归模型
model.fit(X_train, Y_train) #训练模型
# print("模型系数:", model.coef_)
# print("模型截距:", model.intercept_)
print("模型函数: Y = {:.2f} * X + {:.2f}".format(model.coef_[0], model.intercept_))

#模型预测
Y_pred = model.predict(X_test) #预测结果
print("预测结果:", Y_pred)
print("误差:", abs(Y_test - Y_pred)/Y_test) #计算误差

#可视化训练结果
plt.scatter(X_train, Y_train, color='blue', label='Training data')
plt.scatter(X_test, Y_test, color='green', label='Testing data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression line')
plt.xlabel('面积')
plt.ylabel('价格')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()