import charset_normalizer

# Taking care of missing data

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

#missing_values_count = nfl_data.isnull().sum()

#nfl_data.dropna()

#subset_nfl_data.fillna(0)

# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
#subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)

# create a new column, date_parsed, with the parsed dates
#landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")

# if data has more than 1 date format
#landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)

#WHEN WE PARCED THEM WE CAN WORK WITH THEM
# # get the day of the month from the date_parsed column
# day_of_month_landslides = landslides['date_parsed'].dt.day
# day_of_month_landslides.head()