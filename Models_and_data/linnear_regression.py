import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

housing_data_path = "./data/housing.csv"
housing_data = pd.read_csv(housing_data_path)

X = housing_data["total_rooms"].values.reshape(-1, 1)
y = housing_data["median_house_value"].values.reshape(-1, 1)

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

house_model = LinearRegression().fit(train_X, train_y)
price_predictions = house_model.predict(val_X)
val_mae = mean_absolute_percentage_error(price_predictions, val_y)

plt.scatter(train_X, train_y, color="red")
plt.plot(train_X, house_model.predict(train_X))
plt.title("Price vs Total rooms")
plt.xlabel("Total Rooms")
plt.ylabel("Price")
plt.show()

print(house_model.coef_)
print(house_model.intercept_)
