import numpy as np
import pandas as pd

# Multiindexing
data = pd.Series(
    np.random.uniform(size=9),
    index=[["a", "a", "a", "b", "b", "c", "c", "d", "d"], [1, 2, 3, 1, 3, 1, 2, 2, 3]],
)
data["b"]
data["b":"c"]
data.loc[["b", "d"]]
# All values with 2 as a second index
data.loc[:, 2]


# For example, you can rearrange this data into a DataFrame using its unstack method
# Columns as first indexes
data.unstack()

frame = pd.DataFrame(
    np.arange(12).reshape((4, 3)),
    index=[["a", "a", "b", "b"], [1, 2, 1, 2]],
    columns=[["Ohio", "Ohio", "Colorado"], ["Green", "Red", "Green"]],
)

# The inverse operation of unstack is stack
data.unstack().stack()
# nlevels to accsess number of levels
frame.index.nlevels

pd.MultiIndex.from_arrays(
    [["Ohio", "Ohio", "Colorado"], ["Green", "Red", "Green"]], names=["state", "color"]
)
# Reordering and Sorting Levels


#  The swaplevel method takes two level numbers or names and returns a new object with the levels interchanged (but the data is otherwise unaltered)
frame.swaplevel("key1", "key2")
# sorts indexes lexicographically
frame.sort_index(level=1)
frame.swaplevel(0, 1).sort_index(level=0)
# Many descriptive and summary statistics on DataFrame and Series have a level option in which you can specify the level you want to aggregate by on a particular axis.
frame.groupby(level="key2").sum()

# DataFrame’s set_index function will create a new DataFrame using one or more of its columns as the index
frame = pd.DataFrame(
    {
        "a": range(7),
        "b": range(7, 0, -1),
        "c": ["one", "one", "one", "two", "two", "two", "two"],
        "d": [0, 1, 2, 0, 1, 2, 3],
    }
)
frame2 = frame.set_index(["c", "d"])
# By default, the columns are removed from the DataFrame, though you can leave them in by passing drop=False to set_index
frame.set_index(["c", "d"], drop=False)
# reset_index, on the other hand, does the opposite of set_index; the hierarchical index levels are moved into the columns
frame2.reset_index()


pd.merge()  # like SQL JOIN
pd.concat()  # stack together

df1 = pd.DataFrame(
    {
        "key": ["b", "b", "a", "c", "a", "a", "b"],
        "data1": pd.Series(range(7), dtype="Int64"),
    }
)
df2 = pd.DataFrame(
    {"key": ["a", "b", "d"], "data2": pd.Series(range(3), dtype="Int64")}
)
# only witch are in 2 arrays
pd.merge(df1, df2)
pd.merge(df1, df2, on="key")
# if names are different in 2 series use left_on and right on
pd.merge(df3, df4, left_on="lkey", right_on="rkey")
#  By default, pandas.merge does an "inner" join
pd.merge(df1, df2, how="outer")
pd.merge(df3, df4, left_on="lkey", right_on="rkey", how="outer")
# To merge with multiple keys, pass a list of column names:
left = pd.DataFrame(
    {
        "key1": ["foo", "foo", "bar"],
        "key2": ["one", "two", "one"],
        "lval": pd.Series([1, 2, 3], dtype="Int64"),
    }
)
right = pd.DataFrame(
    {
        "key1": ["foo", "foo", "bar", "bar"],
        "key2": ["one", "one", "one", "two"],
        "rval": pd.Series([4, 5, 6, 7], dtype="Int64"),
    }
)
pd.merge(left, right, on=["key1", "key2"], how="outer")
# suffixes to append column names
pd.merge(left, right, on="key1", suffixes=("_left", "_right"))

# Merging on Index on right df and key column in left df
pd.merge(left1, right1, left_on="key", right_index=True)

# to merge on 2 columns simultaneously
pd.merge(lefth, righth, left_on=["key1", "key2"], right_index=True)
# DataFrame has a join instance method to simplify merging by index.
left2.join(
    right2, how="outer"
)  # join method performs a left join on the join keys by default
# Lastly, for simple index-on-index merges, you can pass a list of DataFrames to join as an alternative to using the more general pandas.concat function
left2.join([right2, another])

# to concat 2 arrays
arr = np.arange(12).reshape((3, 4))
np.concatenate([arr, arr], axis=1)

# concatenates arrays by defaul leaving 1 column, but with axis columns - creates separate columns for each array
pd.concat([s1, s2, s3], axis="columns")
# keys will make multiindex key for each df in concat prosess
# In the case of combining Series along axis="columns", the keys become the DataFrame column headers
result = pd.concat([s1, s1, s3], keys=["one", "two", "three"])
# For example, we can name the created axis levels with the names argument
pd.concat(
    [df1, df2], axis="columns", keys=["level1", "level2"], names=["upper", "lower"]
)
# ignore_index=True discards the indexes from each DataFrame and concatenates the data in the columns only, assigning a new default index
pd.concat([df1, df2], ignore_index=True)

# Using numpy.where does not check whether the index labels are aligned or not (and does not even require the objects to be the same length),
# so if you want to line up values by index, use the Series combine_first method
a.combine_first(b)
# The output of combine_first with DataFrame objects will have the union of all the column names
# By default, the innermost level is unstacked (same with stack) You can unstack a different level by passing a level number or name
result.unstack(level=0)
# When you unstack in a DataFrame, the level unstacked becomes the lowest level in the result
data = pd.DataFrame(
    np.arange(6).reshape((2, 3)),
    index=pd.Index(["Ohio", "Colorado"], name="state"),
    columns=pd.Index(["one", "two", "three"], name="number"),
)
result = data.stack()
df = pd.DataFrame(
    {"left": result, "right": result + 5},
    columns=pd.Index(["left", "right"], name="side"),
)
df.unstack(level="state")

# forming period index with columns
# pop returns column and simultaneously delets in from df
periods = pd.PeriodIndex(
    year=data.pop("year"), quarter=data.pop("quarter"), name="date"
)
# converting them to timestamps
data.index = periods.to_timestamp("D")

# forms new table with date as index columns from unique values from columns and values from values (forms 1 date to many data points)
pivoted = long_data.pivot(index="date", columns="item", values="value")
# if original df has more then 1 column pivoted df will have hierarchical columns if no values attribute specifide

# An inverse operation to pivot for DataFrames is pandas.melt
df = pd.DataFrame(
    {"key": ["foo", "bar", "baz"], "A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}
)
melted = pd.melt(df, id_vars="key")
# You can also specify a subset of columns to use as value columns
pd.melt(df, id_vars="key", value_vars=["A", "B"])
# pandas.melt can be used without any group identifiers, too.
pd.melt(df, value_vars=["A", "B", "C"])
