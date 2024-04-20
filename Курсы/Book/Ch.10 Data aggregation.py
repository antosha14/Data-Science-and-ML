import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm  # for econometrics

# 0 - столбик, 1 строка
# DEFAULT  ACROSS THE ROWS (for column), axis columns = ACROSS THE COLUMNS = for rows

df = pd.DataFrame(
    {
        "key1": ["a", "a", None, "b", "b", "a", None],
        "key2": pd.Series([1, 2, 1, 2, 1, None, 1], dtype="Int64"),
        "data1": np.random.standard_normal(7),
        "data2": np.random.standard_normal(7),
    }
)

# 1) Creating group object (groups by key1)
grouped = df["data1"].groupby(df["key1"])
# 2) using operation on group object
grouped.mean()

# grouping with hierarhical index
means = df["data1"].groupby([df["key1"], df["key2"]]).mean()

# numeric_only deletes non numeric columns from the output
df.groupby("key2").mean(numeric_only=True)

# .size() to return group size
df.groupby(["key1", "key2"]).size()
# dropna to control NA values (True by default)
df.groupby("key1", dropna=False).size()
df.groupby(["key1", "key2"], dropna=False).size()

# count counts nonnull values (even when key is null)
df.groupby("key1").count()

# group object supports iteration
for name, group in df.groupby("key1"):
    print(name)
    print(group)


# if grouping is done with multiple keys, keys will be located in a touple
for (k1, k2), group in df.groupby(["key1", "key2"]):
    print((k1, k2))
    print(group)

# one line example
pieces = {name: group for name, group in df.groupby("key1")}
# axis="index" by default, can change to columns3
# example of groupping columns by name
grouped = df.groupby(
    {"key1": "key", "key2": "key", "data1": "data", "data2": "data"}, axis="columns"
)
df

for group_key, group_values in grouped:
    print(group_key)
    print(group_values)

# Indexing a GroupBy object created from a DataFrame with a column name or array of column names has the effect of column subsetting for aggregation.
# this are similar
df.groupby("key1")["data1"]
df.groupby("key1")[["data2"]]

df["data1"].groupby(df["key1"])
df[["data2"]].groupby(df["key1"])
df.groupby(["key1", "key2"])[["data2"]].mean()


people = pd.DataFrame(
    np.random.standard_normal((5, 5)),
    columns=["a", "b", "c", "d", "e"],
    index=["Joe", "Steve", "Wanda", "Jill", "Trey"],
)

people.iloc[2:3, [1, 2]] = np.nan
# column grouping using dictionary
# mapping can also be a series
mapping = {"a": "red", "b": "red", "c": "blue", "d": "blue", "e": "red", "f": "orange"}
by_column = people.groupby(mapping, axis="columns")
by_column.sum()
people

# we can group values by functions
# len is used with every index and groups are formed of names with similar length
people.groupby(len).sum()
# we can mix functions, arrays and dictionaries
key_list = ["one", "one", "one", "two", "two"]
# if we pass a list, its considered as list of keys for every row and then grouping is done by it
people.groupby([len, key_list]).min()

# To group by level, pass the level number or name using the level keyword
columns = pd.MultiIndex.from_arrays(
    [["US", "US", "US", "JP", "JP"], [1, 3, 5, 1, 3]], names=["cty", "tenor"]
)
hier_df = pd.DataFrame(np.random.standard_normal((4, 5)), columns=columns)
hier_df.groupby(level="cty", axis="columns").count()

# we can call Series methods on grouped object
# for example nsmallest (returns spesified number of smallest elements from the series)
df
grouped = df.groupby("key1")
grouped["data1"].nsmallest(2)


# To use your own aggregation functions, pass any function that aggregates an array to the aggregate method or its short alias agg
def peak_to_peak(arr):
    return arr.max() - arr.min()


grouped.agg(peak_to_peak)
# If you pass a list of functions or function names instead, you get back a DataFrame with column names taken from the functions
grouped_pct.agg(["mean", "std", peak_to_peak])
# If you pass a list of (name, function) tuples, the first element of each tuple will be used as the DataFrame column names (you can think of a list of 2-tuples as an ordered mapping)
grouped_pct.agg([("average", "mean"), ("stdev", np.std)])

# Same statistics for multiple columns
functions = ["count", "mean", "max"]
result = grouped[["tip_pct", "total_bill"]].agg(functions)

# different functions to one or more of the columns
grouped.agg({"tip": np.max, "size": "sum"})
grouped.agg({"tip_pct": ["min", "max", "mean", "std"], "size": "sum"})
# as_index = False makes keys return as ordinary columns MORE OPTIMIZED THEN reset_index()
grouped = tips.groupby(["day", "smoker"], as_index=False)


def top(df, n=5, column="tip_pct"):
    return df.sort_values(column, ascending=False)[:n]


top(tips, n=6)

# First, the tips DataFrame is split into groups based on the value of smoker. Then the top function is called on each group, and the results of each function call are glued
# together using pandas.concat, labeling the pieces with the group names
tips.groupby("smoker").apply(top)
# If you pass a function to apply that takes other arguments or keywords, you can pass these after the function
tips.groupby(["smoker", "day"]).apply(top, n=1, column="total_bill")


# Inside GroupBy, when you invoke a method like describe, it is actually just a shortcut for
def f(group):
    return group.describe()


grouped.apply(f)
# to not rewamp indexes pass group_keys= False
tips.groupby("smoker", group_keys=False).apply(top)


frame = pd.DataFrame(
    {"data1": np.random.standard_normal(1000), "data2": np.random.standard_normal(1000)}
)
quartiles = pd.cut(frame["data1"], 4)


# The Categorical object returned by cut can be passed directly to groupby. So we could compute a set of group statistics for the quartiles, like so
def get_stats(group):
    return pd.DataFrame(
        {
            "min": group.min(),
            "max": group.max(),
            "count": group.count(),
            "mean": group.mean(),
        }
    )


grouped = frame.groupby(quartiles)
grouped.apply(get_stats)
# same result as with
grouped.agg(["min", "max", "count", "mean"])

states = [
    "Ohio",
    "New York",
    "Vermont",
    "Florida",
    "Oregon",
    "Nevada",
    "California",
    "Idaho",
]

group_key = ["East", "East", "East", "East", "West", "West", "West", "West"]

data = pd.Series(np.random.standard_normal(8), index=states)
data[["Vermont", "Nevada", "Idaho"]] = np.nan
data.groupby(group_key).size()
data.groupby(group_key).count()
data.groupby(group_key).mean()


# Filling NA values using group means
def fill_mean(group):
    return group.fillna(group.mean())


data.groupby(group_key).apply(fill_mean)

# To fill each colomn depending on its name
fill_values = {"East": 0.5, "West": -1}


def fill_func(group):
    return group.fillna(fill_values[group.name])


data.groupby(group_key).apply(fill_func)


suits = ["H", "S", "C", "D"]  # Hearts, Spades, Clubs, Diamonds
card_val = (list(range(1, 11)) + [10] * 3) * 4
base_names = ["A"] + list(range(2, 11)) + ["J", "K", "Q"]
cards = []
for suit in suits:
    cards.extend(str(num) + suit for num in base_names)

deck = pd.Series(card_val, index=cards)


# to draw a sample of size n use sample()
def draw(deck, n=5):
    return deck.sample(n)


draw(deck)


def get_suit(card):
    # last letter is suit
    return card[-1]


# grouping by last letter and drawing a sample
deck.groupby(get_suit).apply(draw, n=2)


# Calculating group weighted average
df = pd.DataFrame(
    {
        "category": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "data": np.random.standard_normal(8),
        "weights": np.random.uniform(size=8),
    }
)
grouped = df.groupby("category")


def get_wavg(group):
    return np.average(group["data"], weights=group["weights"])


grouped.apply(get_wavg)


# groupwise yearly correlations of dayly returns
def spx_corr(group):
    return group.corrwith(group["SPX"])


rets = close_px.pct_change().dropna()


def get_year(x):
    return x.year


by_year = rets.groupby(get_year)
by_year.apply(spx_corr)


# intercolumn correlation
def corr_aapl_msft(group):
    return group["AAPL"].corr(group["MSFT"])


# executes ordinary least squares (OLS) regression on each chunk of data:
def regress(data, yvar=None, xvars=None):
    Y = data[yvar]
    X = data[xvars]
    X["intercept"] = 1.0
    result = sm.OLS(Y, X).fit()
    return result.params


# yearly linear regression of aapl and microsoft
by_year.apply(regress, yvar="AAPL", xvars=["SPX"])

# transform is similar to apply but imposes more constraints
# 1) It can produce a scalar value to be broadcast to the shape of the group.
# 2) It can produce an object of the same shape as the input group.
# 3) It must not mutate its input.

df = pd.DataFrame({"key": ["a", "b", "c"] * 4, "value": np.arange(12.0)})


def get_mean(group):
    return group.mean()


g = df.groupby("key")["value"]
g.transform(get_mean)
# For built-in aggregation functions, we can pass a string alias as with the GroupBy agg method
g.transform("mean")


# Like apply, transform works with functions that return Series, but the result must be the same size as the input. For example, we can multiply each group by 2 using a helper function:
def times_two(group):
    return group * 2


g.transform(times_two)


# As a more complicated example, we can compute the ranks in descending order for each group
def get_ranks(group):
    return group.rank(ascending=False)


g.transform(get_ranks)


# normalization
def normalize(x):
    return (x - x.mean()) / x.std()


g.transform(normalize)
g.apply(normalize)
normalized = (df["value"] - g.transform("mean")) / g.transform("std")

# PIVOT TABLES
# Returning to the tipping dataset, suppose you wanted to compute a table of group means (the default pivot_table aggregation type) arranged by day and smoker on the rows:
tips.pivot_table(
    index=["day", "smoker"], values=["size", "tip", "tip_pct", "total_bill"]
)
tips.pivot_table(index=["time", "day"], columns="smoker", values=["tip_pct", "size"])
# margins= True to add subtotals
tips.pivot_table(
    index=["time", "day"], columns="smoker", values=["tip_pct", "size"], margins=True
)

# to use other function then mean use aggfunc
tips.pivot_table(
    index=["time", "smoker"], columns="day", values="tip_pct", aggfunc=len, margins=True
)
# If some combinations are empty (or otherwise NA), you may wish to pass a fill_value

tips.pivot_table(
    index=["time", "size", "smoker"], columns="day", values="tip_pct", fill_value=0
)
# A cross-tabulation (or crosstab for short) is a special case of a pivot table that computes group frequencies.

from io import StringIO

data = """Sample  Nationality  Handedness
    1   USA  Right-handed
    2   Japan    Left-handed
    3   USA  Right-handed
    4   Japan    Right-handed
    5   Japan    Left-handed
    6   Japan    Right-handed
    7   USA  Right-handed
    8   USA  Left-handed
    9   Japan    Right-handed
    10  USA  Right-handed"""

data = pd.read_table(StringIO(data), sep="\s+")
pd.crosstab(data["Nationality"], data["Handedness"], margins=True)
pd.crosstab([tips["time"], tips["day"]], tips["smoker"], margins=True)
