import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import os
#获取目录并读取数据集文件，获得一个Dataframe对象
formerpath = os.getcwd()
path= formerpath+"\\PSCompPars_2025.03.06_06.36.39.csv"
data = pd.read_csv(path, comment='#')

#选取需要的列里的数据，去掉缺失值
data = data[['pl_orbsmax','st_mass','pl_orbper']].dropna()
#选择需要的列作为X和y,即输入特征和目标变量
X = data[['pl_orbsmax','st_mass']]
y = data['pl_orbper']
#对数据取对数
y = y.apply(np.log)
X = X.apply(np.log)
'''print(X.head())
print(y.head())'''
#分割训练集和测试集，比例为8:2，random_state用于使随机数每次生成是一致的，确保实验结果可复现
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#创建这个线性回归模型并训练模型
model_1 = LinearRegression()
model_1.fit(X_train, y_train)
#预测测试集中的数据
y_pred = model_1.predict(X_test)
#计算均方误差和R2，评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:",mse,"\nR2:",r2,'\n')
#设置坐标轴最大显示的值
'''plt.xlim(0,20)
plt.ylim(0,35)'''
#绘制散点
plt.scatter(y_test, y_pred, color="blue", label="predict vs.true")
#绘制y=x参考线
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="ideal")
#设置横纵坐标的标签，还有标题
plt.xlabel("T_true")
plt.ylabel("T_predict")
plt.title("predict vs true")
#显示图例
plt.legend(loc="upper left")
#显示图像
plt.show()