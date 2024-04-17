import charset_normalizer
# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling


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


# look at the first ten thousand bytes to guess the character encoding
# with open("../input/kickstarter-projects/ks-projects-201801.csv", 'rb') as rawdata:
#     result = charset_normalizer.detect(rawdata.read(10000))

# # check what the character encoding might be
# print(result)



#FIXING inconsistensy like Germany and gerMany 
# # convert to lower case
# professors['Country'] = professors['Country'].str.lower()
# # remove trailing white spaces
# professors['Country'] = professors['Country'].str.strip()

#deal with spaces. We do use 3rd party package fuzzywuzzy

# # get the top 10 closest matches to "south korea"
# matches = fuzzywuzzy.process.extract("south korea", countries, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio)


#SCALING
# # generate 1000 data points randomly drawn from an exponential distribution
# original_data = np.random.exponential(size=1000)

# # mix-max scale the data between 0 and 1
# scaled_data = minmax_scaling(original_data, columns=[0])

# # plot both together to compare
# fig, ax = plt.subplots(1, 2, figsize=(15, 3))
# sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
# ax[0].set_title("Original Data")
# sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
# ax[1].set_title("Scaled data")
# plt.show()