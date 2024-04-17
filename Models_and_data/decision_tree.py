import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

housing_data_path = "./data/housing.csv"
housing_data = pd.read_csv(housing_data_path)

housing_data.head()