# 0 - столбик, 1 строка
# DEFAULT  ACROSS THE ROWS (for column), axis columns = ACROSS THE COLUMNS = for rows
from datetime import datetime

import numpy as np
import pandas as pd
import pytz  # library for time zones
from pandas.tseries.frequencies import to_offset
from pandas.tseries.offsets import Hour, Minute

now = datetime.now()
now.year, now.month, now.day

stamp = pd.Timestamp("2011-03-12 04:00")  # MOMENT IN TIME
p = pd.Period("2011", freq="A-DEC")  # TIME SPAN, represents whole year

# timedelta represents differene
delta = datetime(2011, 1, 7) - datetime(2008, 6, 24, 8, 15)

# we can convert datetime to string
stamp = datetime(2011, 1, 3)
str(stamp)
stamp.strftime("%Y-%m-%d")

# we use datetime.strptime() to convert string to date
value = "2011-01-03"
datetime.strptime(value, "%Y-%m-%d")

# pd.to_datetime also parses popular datetime strings and returns DatetimeIndex
datestrs = ["2011-07-06 12:00:00", "2011-08-06 00:00:00"]
pd.to_datetime(datestrs)
# it also considers Nan as Nat
idx = pd.to_datetime(datestrs + [None])

# pandas.Timestamp  stores nanoseconds, np.datetime64 doesnt
# For longer time series, a year or only a year and month can be passed to easily select slices of data
longer_ts = pd.Series(
    np.random.standard_normal(1000), index=pd.date_range("2000-01-01", periods=1000)
)
# can filter by year passing string
longer_ts["2001"]
# also works for month
longer_ts["2001-05"]
# Slicing with datetime objects works as well
ts[datetime(2011, 1, 7) :]
ts[datetime(2011, 1, 7) : datetime(2011, 1, 10)]
# Because most time series data is ordered chronologically, you can slice with timestamps not contained in a time series to perform a range query
ts["2011-01-06":"2011-01-11"]  # VIEWS
ts.truncate(after="2011-01-09")

# Suppose you wanted to aggregate the data having nonunique timestamps. One way to do this is to use groupby and pass level=0 (the one and only level)
grouped = dup_ts.groupby(level=0)

# or example, you can convert the sample time series to fixed daily frequency by calling resample
resampler = ts.resample("D")
# While I used it previously without explanation, pandas.date_range is responsible for generating a DatetimeIndex with an indicated length according to a particular frequency
index = pd.date_range("2012-04-01", "2012-06-01")
# By default, pandas.date_range generates daily timestamps. If you pass only a start or end date, you must pass a number of periods to generate:
pd.date_range(start="2012-04-01", periods=20)
# end of business month
pd.date_range("2000-01-01", "2000-12-01", freq="BM")

# Base time series frequencies (not comprehensive)
pd.date_range("2012-05-02 12:56:31", periods=5, normalize=True)
# Sometimes you will have start or end dates with time information but want to generate a set of timestamps normalized to midnight as a convention. To do this, there is a normalize option

# creates DatetimeIndex with 4 hours delimeter
pd.date_range("2000-01-01", "2000-01-03 23:59", freq="4H")
# Offsets can be combined by + sighn
Hour(2) + Minute(30)
# Similarly, you can pass frequency strings, like "1h30min", that will effectively be parsed to the same expression
pd.date_range("2000-01-01", periods=10, freq="1h30min")

# 3rd friday of each month
monthly_dates = pd.date_range("2012-01-01", "2012-09-01", freq="WOM-3FRI")
list(monthly_dates)


# Shifting refers to moving data backward and forward through time. Both Series and DataFrame have a shift method for
# doing naive shifts forward or backward, leaving the index unmodified
# shift moves data forwards in time by specified positions
# introduces Nan values
ts = pd.Series(
    np.random.standard_normal(4), index=pd.date_range("2000-01-01", periods=4, freq="M")
)
ts.shift(2)
ts.shift(-2)

# A common use of shift is computing consecutive percent changes in a time series or multiple time series as DataFrame columns. This is expressed as
ts / ts.shift(1) - 1
# Because naive shifts leave the index unmodified, some data is discarded. Thus if the frequency is known, it can be passed to shift to advance the timestamps instead of simply the data
ts.shift(2, freq="M")
ts.shift(3, freq="D")
ts.shift(1, freq="90T")


from pandas.tseries.offsets import Day, MonthEnd

now = datetime(2011, 11, 17)
now + 3 * Day()

# If you add an anchored offset like MonthEnd, the first increment will "roll forward" a date to the next date according to the frequency rule
now + MonthEnd()
now + MonthEnd(2)

offset.rollforward(now)
offset.rollback(now)

ts = pd.Series(
    np.random.standard_normal(20),
    index=pd.date_range("2000-01-15", periods=20, freq="4D"),
)
ts.groupby(MonthEnd().rollforward).mean()
# same as
ts.resample("M").mean()


pytz.common_timezones[-5:]
# To get a time zone object from pytz, use pytz.timezone
tz = pytz.timezone("America/New_York")
# Methods in pandas will accept either time zone names or these objects


# By default, time series in pandas are time zone naive. For example, consider the following time series
dates = pd.date_range("2012-03-09 09:30", periods=6)
ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
print(ts.index.tz)  # by default is None

pd.date_range("2012-03-09 09:30", periods=10, tz="UTC")
# Conversion from naive to localized (reinterpreted as having been observed in a particular time zone) is handled by the tz_localize method:
ts
ts_utc = ts.tz_localize("UTC")
# Once a time series has been localized to a particular time zone, it can be converted to another time zone with tz_convert
ts_utc.tz_convert("America/New_York")

# In the case of the preceding time series, which straddles a DST transition in the America/New_York time zone, we could localize
# to US Eastern time and convert to, say, UTC or Berlin time
ts_eastern = ts.tz_localize("America/New_York")
ts_eastern.tz_convert("UTC")
ts_eastern.tz_convert("Europe/Berlin")

# Same using pandas
stamp = pd.Timestamp("2011-03-12 04:00")
stamp_utc = stamp.tz_localize("utc")
stamp_utc.tz_convert("America/New_York")
stamp_moscow = pd.Timestamp("2011-03-12 04:00", tz="Europe/Moscow")

# Time zone-aware Timestamp objects internally store a UTC timestamp value as nanoseconds since the Unix epoch (January 1, 1970),
# so changing the time zone does not alter the internal UTC value

# data offset сумма учитывает переход на летнее время
stamp = pd.Timestamp("2012-03-11 01:30", tz="US/Eastern")
stamp + Hour()
# If two time series with different time zones are combined, the result will be UTC
dates = pd.date_range("2012-03-07 09:30", periods=10, freq="B")
ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
ts1 = ts[:7].tz_localize("Europe/London")
ts2 = ts1[2:].tz_convert("Europe/Moscow")
result = ts1 + ts2
result.index
# Operations between time zone-naive and time zone-aware data are not supported and will raise an exception.

# Conveniently, adding and subtracting integers from periods has the effect of shifting their frequency
p = pd.Period("2011", freq="A-DEC")  # TIME SPAN
p + 5
p - 2

# If two periods have the same frequency, their difference is the number of units between them as a date offset
pd.Period("2014", freq="A-DEC") - p
# Regular ranges of periods can be constructed with the period_range function

# Regular ranges of periods can be constructed with the period_range function
periods = pd.period_range("2000-01-01", "2000-06-30", freq="M")

values = ["2001Q3", "2002Q2", "2003Q1"]
index = pd.PeriodIndex(values, freq="Q-DEC")

# convert yearly frequence to monthly
p = pd.Period("2011", freq="A-DEC")
p.asfreq("M", how="start")
p.asfreq("M", how="end")

# You can think of Period("2011", "A-DEC") as being a sort of cursor pointing to a span of time
p = pd.Period("Aug-2011", "M")
p.asfreq("A-JUN")
ts.asfreq("B", how="end")

# Thus, the period 2012Q4 has a different meaning depending on fiscal year end. pandas supports all 12 possible quarterly frequencies as Q-JAN through Q-DEC
p = pd.Period("2012Q4", freq="Q-JAN")
# In the case of a fiscal year ending in January, 2012Q4 runs from November 2011 through January 2012, which you can check by converting to daily frequency
p.asfreq("D", how="start")
# The to_timestamp method returns the Timestamp at the start of the period by default.
# You can generate quarterly ranges using pandas.period_range. The arithmetic is identical, too
new_periods = (periods.asfreq("B", "end") - 1).asfreq("H", "start") + 16
ts.index = new_periods.to_timestamp()


dates = pd.date_range("2000-01-01", periods=3, freq="M")
ts = pd.Series(np.random.standard_normal(3), index=dates)
pts = ts.to_period()

dates = pd.date_range("2000-01-29", periods=6)
ts2 = pd.Series(np.random.standard_normal(6), index=dates)
ts2.to_period("M")
pts = ts2.to_period()
pts.to_timestamp(how="end")

# to form periods from years and quaters in different columns in DF
index = pd.PeriodIndex(year=data["year"], quarter=data["quarter"], freq="Q-DEC")

# pandas objects are equipped with a resample method, which is the workhorse function for all frequency conversion. resample has a similar API to groupby;
# you call resample to group the data, then call an aggregation function
dates = pd.date_range("2000-01-01", periods=100)
ts = pd.Series(np.random.standard_normal(len(dates)), index=dates)
ts.resample("M").mean()
ts.resample("M", kind="period").mean()

ts.resample("5min").sum()
# The frequency you pass defines bin edges in five-minute increments. For this frequency, by default the left bin edge is inclusive, so the 00:00 value is included in the 00:00 to 00:05 interval, and the 00:05 value is excluded from that interval
# to change lables and inclusion
ts.resample("5min", closed="right", label="right").sum()
result = ts.resample("5min", closed="right", label="right").sum()
result.index = result.index + to_offset("-1s")
ts = pd.Series(np.random.permutation(np.arange(len(dates))), index=dates)
# to return open high and close data for a period
ts.resample("5min").ohlc()


# Upsampling is converting from a lower frequency to a higher frequency, where no aggregation is needed. Let’s consider a DataFrame with some weekly data:
# When you are using an aggregation function with this data, there is only one value per group, and missing values result in the gaps. We use the asfreq method to convert to the higher frequency without any aggregation
frame = pd.DataFrame(
    np.random.standard_normal((2, 4)),
    index=pd.date_range("2000-01-01", periods=2, freq="W-WED"),
    columns=["Colorado", "Texas", "New York", "Ohio"],
)
df_daily = frame.resample("D").asfreq()
# The same filling or interpolation methods available in the fillna and reindex methods are available for resampling
frame.resample("D").ffill()
# You can similarly choose to only fill a certain number of periods forward to limit how far to continue using an observed value
frame.resample("D").ffill(limit=2)

df2 = pd.DataFrame(
    {
        "time": times.repeat(3),
        "key": np.tile(["a", "b", "c"], N),
        "value": np.arange(N * 3.0),
    }
)
# To do the same resampling for each value of "key", we introduce the pandas.Grouper object
time_key = pd.Grouper(freq="5min")
# We can then set the time index, group by "key" and time_key, and aggregate
resampled = df2.set_index("time").groupby(["key", time_key]).sum()
resampled.reset_index()


close_px_all = pd.read_csv("./Data/stock_px.csv", parse_dates=True, index_col=0)
close_px = close_px_all[["AAPL", "MSFT", "XOM"]]
close_px = close_px.resample("B").ffill()


# I now introduce the rolling operator, which behaves similarly to resample and groupby. It can be called on a Series or DataFrame along with a window (expressed as a number of periods; see Apple price with 250-day moving average for the plot created)
close_px["AAPL"].plot()
close_px["AAPL"].rolling(250).mean().plot()
# The expression rolling(250) is similar in behavior to groupby, but instead of grouping, it creates an object that enables grouping over a 250-day sliding window. So here we have the 250-day moving window average of Apple's stock price

#
std250 = close_px["AAPL"].pct_change().rolling(250, min_periods=10).std()
# To compute an expanding window mean, use the expanding operator instead of rolling. The expanding mean starts the time window from the same point as the rolling window and increases the size of the window until it encompasses the whole series. An expanding window mean on the std250 time series looks like this
expanding_mean = std250.expanding().mean()
# Calling a moving window function on a DataFrame applies the transformation to each column

close_px.rolling(60).mean().plot(logy=True)
# 20 days rolling mean
close_px.rolling("20D").mean()

# Here’s an example comparing a 30-day moving average of Apple’s stock price with an exponentially weighted (EW) moving average with span=60
aapl_px = close_px["AAPL"]["2006":"2007"]
ma30 = aapl_px.rolling(30, min_periods=20).mean()
ewma30 = aapl_px.ewm(span=30).mean()
aapl_px.plot(style="k-", label="Price")
ma30.plot(style="k--", label="Simple Moving Avg")
ewma30.plot(style="k-", label="EW MA")

# CORRELATION (ROLLING)
spx_px = close_px_all["SPX"]
spx_rets = spx_px.pct_change()
returns = close_px.pct_change()
corr = returns["AAPL"].rolling(125, min_periods=100).corr(spx_rets)
# Suppose you wanted to compute the rolling correlation of the S&P 500 index with many stocks at once. You could write a loop computing this for each stock like we did for Apple above, but if each stock is a column in a single DataFrame, we can compute all of the rolling correlations in one shot by calling rolling on the DataFrame and passing the spx_rets Series
corr = returns.rolling(125, min_periods=100).corr(spx_rets)
# The apply method on rolling and related methods provides a way to apply an array function of your own creation over a moving window. The only requirement is that the function produce a single value (a reduction) from each piece of the array. For example, while we can compute sample quantiles using rolling(...).quantile(q), we might be interested in the percentile rank of a particular value over the sample. The scipy.stats.percentileofscore function does just this
from scipy.stats import percentileofscore


def score_at_2percent(x):
    return percentileofscore(x, 0.02)


result = returns["AAPL"].rolling(250).apply(score_at_2percent)
