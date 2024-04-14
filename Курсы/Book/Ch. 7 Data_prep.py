import numpy as np
import pandas as pd

# to filter missing data we can use data.notna() as boolean index or use .dropna()
data = pd.Series([1, np.nan, 3.5, np.nan, 7])
data.dropna()
data[data.notna()]

# dropna by default drops any row containing a missing value
# passing how="all" will drop only rows that are all NA
# RETERNS NEW OBJECT
data.dropna(how="all")

# to drop columns pass axis
data.dropna(axis="columns", how="all")


df = pd.DataFrame(np.random.standard_normal((7, 3)))
# Suppose you want to keep only rows containing at most a certain number of missing observations. You can indicate this with the thresh argument
df.dropna(thresh=2)

# to fill NA with values
# Calling fillna with a dictionary, you can use a different fill value for each column
df.fillna(0)
data.fillna(data.mean())

# remove duplicates
# The DataFrame method duplicated returns a Boolean Series indicating whether each row is
# a duplicate (its column values are exactly equal to those in an earlier row) or not
# CONSIDERS ALL OF THE COLUMNS
data = pd.DataFrame({"k1": ["one", "two"] * 3 + ["two"], "k2": [1, 1, 2, 3, 3, 4, 4]})
data.duplicated()


# Relatevly, drop_duplicates returns a DataFrame with
# rows where the duplicated array is False filtered out
data.drop_duplicates()

# to filter ALL duplicates in one column
data.drop_duplicates(subset=["k1"])
# duplicated and drop_duplicates by default keep the first observed value combination.
# Passing keep="last" will return the last one
data.drop_duplicates(["k1", "k2"], keep="last")


# Map method is used to create mapping from 1 column to another using dict or function
meat_to_animal = {
    "bacon": "pig",
    "pulled pork": "pig",
    "pastrami": "cow",
    "corned beef": "cow",
    "honey ham": "pig",
    "nova lox": "salmon",
}

# we could use map to do element wise transformations
data["animal"] = data["food"].map(meat_to_animal)


# same as
def get_animal(x):
    return meat_to_animal[x]


data["food"].map(get_animal)

# replacement
data = pd.Series([1.0, -999.0, 2.0, -999.0, -1000.0, 3.0])
# substitute 1 with 1, 2nd with 2nd
data.replace([-999, -1000], [np.nan, 0])
# same as
data.replace({-999: np.nan, -1000: 0})


# We can transform indexes and column names using map as well
data = pd.DataFrame(
    np.arange(12).reshape((3, 4)),
    index=["Ohio", "Colorado", "New York"],
    columns=["one", "two", "three", "four"],
)


def transform(x):
    return x[:4].upper()


data.index.map(transform)

# If you want to create a transformed version of a dataset without
# modifying the original, a useful method is rename
# rename saves you from the chore of copying the DataFrame manually and assigning new values to its index and columns attributes
data.rename(index=str.title, columns=str.upper)
data.rename(index={"OHIO": "INDIANA"}, columns={"three": "peekaboo"})

# To group by values into buckets
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]
age_categories = pd.cut(ages, bins)

# to view group codes and diapasons
age_categories.categories
age_categories.codes

pd.value_counts(age_categories)

# We can override the default interval-based bin labeling by
# passing a list or array to the labels option

group_names = ["Youth", "YoungAdult", "MiddleAged", "Senior"]
pd.cut(ages, bins, labels=group_names)

# If you pass an integer number of bins to pandas.cut instead of explicit bin edges, it will
# compute equal-length bins based on the minimum and maximum values in the data
data = np.random.uniform(size=20)
pd.cut(data, 4, precision=2)  # precision to limit decimal precision


# to slice by quartiles use pd.qcut()
data = np.random.standard_normal(1000)
quartiles = pd.qcut(data, 4, precision=2)
pd.value_counts(quartiles)

# Similar to pandas.cut, you can pass your own quantiles (numbers between 0 and 1, inclusive)
pd.qcut(data, [0, 0.1, 0.5, 0.9, 1.0]).value_counts()

data = pd.DataFrame(np.random.standard_normal((1000, 4)))
col = data[2]
col[col.abs() > 3]
# any Returns whether any element is True, potentially over an axis.
data[(data.abs() > 3).any(axis="columns")]
# np.sign to get signs of every values
data[data.abs() > 3] = np.sign(data) * 3

# Permuting (randomly reordering) a Series or the rows in a DataFrame is possible using the numpy.random.permutation function. Calling permutation with the length
# of the axis you want to permute produces an array of integers indicating the new ordering
df = pd.DataFrame(np.arange(100 * 7).reshape((100, 7)))
sampler = np.random.permutation(100)
df.take(sampler)
df.iloc[sampler]
# By invoking take with axis="columns", we could also select a permutation of the columns
df.take(sampler, axis="columns")
# To select a random subset without replacement (the same row cannot appear twice), you can use the sample method on Series and DataFrame
df.sample(n=3)
# To generate a sample with replacement (to allow repeat choices), pass replace=True to sample
choices = pd.Series([5, 7, -1, 6, 4])
choices.sample(n=10, replace=True)


# Computing Indicator/Dummy Variables
# Creates Dummy variables for categories
df = pd.DataFrame({"key": ["b", "b", "a", "c", "a", "b"], "data1": range(6)})
# we can add prefix before each column name
dummies = pd.get_dummies(df["key"], prefix="key", dtype=float)
df_with_dummy = df[["data1"]].join(dummies)

# pandas has implemented a special Series method str.get_dummies
# that handles this scenario of multiple group membership encoded as a delimited string
dummies = movies["genres"].str.get_dummies("|")
movies_windic = movies.join(dummies.add_prefix("Genre_"))
# A useful recipe for statistical applications is to combine pandas.get_dummies with a discretization function like pandas.cut
pd.get_dummies(pd.cut(values, bins))

# by default pandas series uses standart nunpy types
s = pd.Series([1, 2, 3, None])
# creating series using new pandas dtype
s = pd.Series([1, 2, 3, None], dtype=pd.Int64Dtype())
s[3] is pd.NA

# pandas also has an extension type specialized for string data that does not
# use NumPy object arrays (it requires the pyarrow library, which you may need to install separately)
s = pd.Series(["one", "two", None, "three"], dtype=pd.StringDtype())
# These string arrays generally use much less memory and are frequently computationally more efficient for doing operations on large datasets
# Extension types can be passed to the Series astype method, allowing you to convert easily as part of your data cleaning process
df = pd.DataFrame(
    {
        "A": [1, 2, None, 4],
        "B": ["one", "two", "three", None],
        "C": [False, None, False, True],
    }
)
df["A"] = df["A"].astype("Int64")
df["B"] = df["B"].astype("string")
df["C"] = df["C"].astype("boolean")
df

pieces = ["a", "b", "guido"]
"::".join(pieces)  # To join strings with a delimiter
str.index("s")  # exception if not found
str.find("s")  # -1 if not found
str.count("s")
str.replace(",", "s")

# Regular expressions
import re

text = "foo    bar\t baz  \tqux"
re.split(r"\s+", text)
["foo", "bar", "baz", "qux"]
# we can compile regex
# Creating a regex object with re.compile is highly recommended if you intend to apply the same expression to many strings; doing so will save CPU cycles.
regex = re.compile(r"\s+")
regex.split(text)
regex.findall(text)  # returns list of all matches


text = """Dave dave@google.com
Steve steve@gmail.com
Rob rob@gmail.com
Ryan ryan@yahoo.com"""
pattern = r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}"

# re.IGNORECASE makes the regex case insensitive
regex = re.compile(pattern, flags=re.IGNORECASE)
regex.findall(text)
m = regex.search(
    text
)  # search returns a special match object for the first email address in the text. For the preceding regex, the match object can only tell us the start and end position of the pattern in the string:
text[m.start() : m.end()]
regex.match(text)  # True if pattern occurs at the start of the string
regex.sub(
    "REDACTED", text
)  # sub will return a new string with occurrences of the pattern replaced by a new string


# if we want to segment string we use () in regex to indicate segment and
# use regex.match(string).groups() to get a touple with segments
pattern = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"
regex = re.compile(pattern, flags=re.IGNORECASE)
m = regex.match("wesm@bright.net")
m.groups()
# findall with groups returns list of touples
regex.findall(text)
# sub also has access to groups in each match using special symbols like \1 and \2.
# The symbol \1 corresponds to the first matched group, \2 corresponds to the second, and so forth:
print(regex.sub(r"Username: \1, Domain: \2, Suffix: \3", text))

# String and regular expression methods can be applied (passing a lambda or other function) to each value using data.map, but it will fail on the NA (null) values.
# series has spesial methods which skip NA values. We can acsess them via series.str.method()
data = {
    "Dave": "dave@google.com",
    "Steve": "steve@gmail.com",
    "Rob": "rob@gmail.com",
    "Wes": np.nan,
}
data = pd.Series(data)
data.str.contains("gmail")
pattern = r"([A-Z0-9._%+-]+)@([A-Z0-9.-]+)\.([A-Z]{2,4})"
data.str.findall(pattern, flags=re.IGNORECASE)
matches = data.str.findall(pattern, flags=re.IGNORECASE).str[0]
# to get spec indexed element from tuples we could use .get method
matches.str.get(1)

# strings also can be sliced (vectorized) with .str[slice]
data.str[:5]

# The str.extract method will return the captured groups of a regular expression as a DataFrame
data.str.extract(pattern, flags=re.IGNORECASE)

# In data warehousing, a best practice is to use so-called dimension tables containing
# the distinct values and storing the primary observations as integer keys referencing the dimension table
values = pd.Series([0, 1, 0, 0] * 2)
dim = pd.Series(["apple", "orange"])
# We can use the take method to restore the original Series of strings
dim.take(values)

# pandas has a special Categorical extension type for holding data that uses the integer-based categorical representation or encoding
fruits = ["apple", "orange", "apple", "apple"] * 2
N = len(fruits)
rng = np.random.default_rng(seed=12345)
df = pd.DataFrame(
    {
        "fruit": fruits,
        "count": rng.integers(3, 15, size=N),
        "basket_id": np.arange(N),
        "weight": rng.uniform(0, 4, size=N),
    },
    columns=["basket_id", "fruit", "count", "weight"],
)

df["fruit"]
# Convert to category dtype
fruit_cat = df["fruit"].astype("category")
# The values for fruit_cat are now an instance of pandas.Categorical, which you can access via the .array attribute
c = fruit_cat.array
# The Categorical object has categories and codes attributes
c.categories
c.codes

# to get mapping between codes and categories
dict(enumerate(c.categories))
# You can also create pandas.Categorical directly from other types of Python sequences
my_categories = pd.Categorical(["foo", "bar", "baz", "foo", "bar"])

# PERFORMANCE IS BETTER WITH CATEGORICALS
# if we have categories and codes we can create Categorical from them
categories = ["foo", "bar", "baz"]
codes = [0, 1, 2, 0, 0, 1]
my_cats_2 = pd.Categorical.from_codes(
    codes, categories, ordered=True
)  # ordered parametr to make order metter (random order by default)
# An unordered categorical instance can be made ordered with as_ordered
# not only strings but any immutable type
my_cats_2.as_ordered()

# we can use labels in qcut to name quartiles
rng = np.random.default_rng(seed=12345)
draws = rng.standard_normal(1000)
bins = pd.qcut(draws, 4)
bins = pd.qcut(draws, 4, labels=["Q1", "Q2", "Q3", "Q4"])
bins.codes[:10]
bins = pd.Series(bins, name="quartile")
# We can use groupby bc bins are labeled
# agg()  Aggregate using one or more operations over the specified axis
results = pd.Series(draws).groupby(bins).agg(["count", "min", "max"]).reset_index()


# Categorical is significantly more performant

N = 10_000_000
labels = pd.Series(["foo", "bar", "baz", "qux"] * (N // 4))
categories = labels.astype("category")
labels.memory_usage(deep=True)
categories.memory_usage(deep=True)

# %timeit labels.value_counts()

# Special categorical methods
s = pd.Series(["a", "b", "c", "d"] * 2)
cat_s = s.astype("category")
# The special accessor attribute cat provides access to categorical methods
cat_s.cat.codes
cat_s.cat.categories
# to set new categories
# While it appears that the data is unchanged, the new categories will be reflected in operations that use them. For example, value_counts respects the categories, if present
actual_categories = ["a", "b", "c", "d", "e"]
cat_s2 = cat_s.cat.set_categories(actual_categories)
cat_s.value_counts()
cat_s2.value_counts()
# we can use the remove_unused_categories method to trim unobserved categories
cat_s3 = cat_s[cat_s.isin(["a", "b"])]
cat_s3.cat.remove_unused_categories()
