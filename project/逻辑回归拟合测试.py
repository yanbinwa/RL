import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# 通过二次线性拟合数据
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.log(X).ravel() + np.random.randn(80) * 0.1

lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, lin_reg.predict(X), color='red', label='Quadratic Fit')
plt.legend()
plt.show()

# 获取系数和截距
coefficient = lin_reg.coef_[0][0]
intercept = lin_reg.intercept_[0]

# 打印回归表达式
print(f"回归表达式：y = {coefficient:.2f}x + {intercept:.2f}")
