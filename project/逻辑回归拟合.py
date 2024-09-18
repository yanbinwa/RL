import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


INPUT_FILE = 'file/result{}.xlsx'
INPUT_TYPE_ALL = "all"
INPUT_TYPE_05 = "05"
INPUT_TYPE_95 = "95"


def fitting(input_type):
    # 通过二次线性拟合数据
    df = pd.read_excel(INPUT_FILE.format(input_type))
    X = df['分数'].values.reshape(-1, 1)
    y = df['生存时间'].values

    # 创建多项式特征转换器，并计算新的特征
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    regressor = LinearRegression()
    regressor.fit(X_poly, y)
    return X, y, regressor, poly_features


def show(X, y, regressor_all, poly_features_all, regressor_05, poly_features_05, regressor_95, poly_features_95):
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X, regressor_all.predict(poly_features_all.transform(X)), color='red', label='avg')
    plt.plot(X, regressor_05.predict(poly_features_05.transform(X)), color='yellow', label='min')
    plt.plot(X, regressor_95.predict(poly_features_95.transform(X)), color='green', label='max')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X, y, regressor_all, poly_features_all = fitting(INPUT_TYPE_ALL)
    _, _, regressor_05, poly_features_05 = fitting(INPUT_TYPE_05)
    _, _, regressor_95, poly_features_95 = fitting(INPUT_TYPE_95)
    show(X, y, regressor_all, poly_features_all, regressor_05, poly_features_05, regressor_95, poly_features_95)