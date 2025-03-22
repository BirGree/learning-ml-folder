#第三题：线性回归，这个文件是第三题的直接调用sklearn库的线性回归模型的实现
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import os

formerpath = os.getcwd()
#设置文件路径
#获取目录并读取数据集文件，获得一个Dataframe对象
path= formerpath+"\\PSCompPars_2025.03.06_06.36.39.csv"
#这是数据集的文件路径
data = pd.read_csv(path, comment='#')
#读取数据集文件，去掉注释行
#pd.read_csv()函数读取csv文件，comment参数的作用是去掉注释行
#以上完成了数据集的读取

data = data[['pl_orbsmax','st_mass','pl_orbper']].dropna()
#选取需要的列里的数据，去掉缺失值
X = data[['pl_orbsmax','st_mass']]
y = data['pl_orbper']
#选择需要的列作为X和y,即输入特征和目标变量

y = y.apply(np.log)
X = X.apply(np.log)
#对数据取对数，我们知道根据开普勒第三定律，周期和半长轴的关系取对数之后是线性关系
'''print(X.head())
print(y.head())'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#分割训练集和测试集，比例为8:2，random_state用于使随机数每次生成是一致的，确保实验结果可复现

model_1 = LinearRegression()
model_1.fit(X_train, y_train)
#创建这个线性回归模型并训练模型

y_pred = model_1.predict(X_test)
#预测测试集中的数据

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:",mse,"\nR2:",r2,'\n')
#计算均方误差和R2，评估模型

'''plt.xlim(0,20)
plt.ylim(0,35)'''
#设置坐标轴最大显示的值，本题最后未使用

plt.scatter(y_test, y_pred, color="blue", label="predict vs.true")
#绘制散点图，x轴为真实值，y轴为预测值，颜色为蓝色，标签为"predict vs.true"
#plt.scatter()函数绘制散点图，color参数设置颜色，label参数设置标签

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="ideal")
#绘制y=x参考线
plt.xlabel("T_true")
plt.ylabel("T_predict")
plt.title("predict vs true")
#设置横纵坐标的标签，还有标题
plt.legend(loc="upper left")
#显示图例
plt.show()
#显示图像