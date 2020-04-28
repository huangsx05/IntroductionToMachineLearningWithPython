import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# load dataset and visualize
ram_prices = pd.read_csv("data/ram_price.csv")
print("Shape of data: {}".format(ram_prices.shape))  
plt.semilogy(ram_prices.date, ram_prices.price) # 半对数的图形
plt.xlabel("Year")
plt.ylabel("Price in $/Mbyte")
plt.show()

## split data
#  use historical data to forecast prices after the year 2000
data_train = ram_prices[ram_prices.date < 2000] # before year of 2000
data_test = ram_prices[ram_prices.date >= 2000]
# predict prices based on date
X_train = data_train.date[:, np.newaxis] # newaxis: 增加一个新维度去变成二维矩阵 (去掉会出错)
# log-transform of y
y_train = np.log(data_train.price)

# train - two models for comparison
tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)

# predict on all data (use all instead of test just for visualization purpose)
X_all = ram_prices.date[:, np.newaxis]
pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

# undo log-transform
price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)

# plot
plt.semilogy(data_train.date, data_train.price, label="Training data")
plt.semilogy(data_test.date, data_test.price, label="Test data")
plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
plt.legend()
plt.show()