import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

housing_data_path = "./data/housing.csv"
housing_data = pd.read_csv(housing_data_path)


y=housing_data['median_house_value']
x=housing_data['total_rooms']

train_X, val_X, train_y, val_y = train_test_split(x,y, random_state=1)
train_X = train_X.values.reshape(-1,1)
val_X = val_X.values.reshape(-1,1)

house_model = LinearRegression().fit(train_X, train_y)
price_predictions = house_model.predict(val_X)
val_mae = mean_absolute_percentage_error(price_predictions,val_y)
print(val_mae)