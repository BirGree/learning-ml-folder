#还是第三题，但是这次不使用sklearn的线性回归模型，而是手动实现最小二乘法的线性回归模型
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import os
formerpath = os.getcwd()
path= formerpath+"\\PSCompPars_2025.03.06_06.36.39.csv"
data = pd.read_csv(path, comment='#')

data = data[['pl_orbsmax','st_mass','pl_orbper']].dropna()

X = data[['pl_orbsmax','st_mass']]
y = data['pl_orbper']
y = y.apply(np.log); X = X.apply(np.log)
'''print(X.head())
print(y.head())'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#以下代码改为手动使用最小二乘法的模型
X_b= np.c_[np.ones((X_train.shape[0], 1)),X_train]
#为自变量矩阵添加截距项1

B= np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
#求逆矩阵并计算B，即系数。@符号表示矩阵乘法，这是numpy库里的功能

X_new= np.c_[np.ones((X_test.shape[0], 1)), X_test]
#获取一个有截距项的测试集

y_pred= X_new @ B
#进行预测

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE:",mse,"\nR2:",r2,'\n')

'''plt.xlim(0,20)
plt.ylim(0,35)'''

plt.scatter(y_test, y_pred, color="blue", label="predict vs.true")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="ideal")
plt.xlabel("T_true")
plt.ylabel("T_predict")
plt.title("predict vs true")

plt.legend(loc="upper left")
plt.show()