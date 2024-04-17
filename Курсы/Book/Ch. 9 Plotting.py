import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 0 - столбик, 1 строка
# DEFAULT  ACROSS THE ROWS (for column), axis columns = ACROSS THE COLUMNS = for rows
# В операциях типа frame.div(np.arange(4), axis="index") работает также только к каждой колонке поэлементно добавляется что-то

data = np.arange(10)
plt.plot(data)

# Plots in matplotlib reside within a Figure object. You can create a new figure with plt.figure
fig = plt.figure(figsize=(19.2, 10.8))
# You can’t make a plot with a blank figure. You have to create one or more subplots using add_subplot
ax1 = fig.add_subplot(3, 3, 1)  # 2x2 plot matrix and selecting 1 plot of them
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax5 = fig.add_subplot(3, 3, 9)
ax5.plot(np.random.standard_normal(50).cumsum(), color="black", linestyle="dashed")
ax1.hist(
    np.random.standard_normal(100), bins=20, color="black", alpha=0.3
)  # alpha sets transparancy
ax2.scatter(np.arange(30), np.arange(30) + 3 * np.random.standard_normal(30))


# returns fig, and n dimentional array of subplots
# sharex and y to make plots use same axis
fig, axes = plt.subplots(2, 3, sharex=True)

# .subplot_adjust is used to create space between charts
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
for i in range(2):
    for j in range(2):
        axes[i, j].hist(
            np.random.standard_normal(500), bins=50, color="black", alpha=0.5
        )
fig.subplots_adjust(wspace=0.1, hspace=0.1)

# ax.plot(x, y, linestyle="--", color="green")
ax = fig.add_subplot()
ax.plot(
    np.random.standard_normal(30).cumsum(),
    color="black",
    linestyle="dashed",
    marker="o",
)
# we can use drawstyle="steps-post", label="steps-post" to not use interpolation
# ax.legend() to add legend
# labels to be able to add labels
fig = plt.figure()
ax = fig.add_subplot()
data = np.random.standard_normal(30).cumsum()
ax.plot(data, color="black", linestyle="dashed", label="Default")
ax.plot(
    data, color="green", linestyle="dashed", drawstyle="steps-post", label="steps-post"
)
ax.legend()


#
#
#
# Ticks, Labels, and Legends
#
#
#
fig, ax = plt.subplots()
ax.plot(np.random.standard_normal(1000).cumsum(), color="black", label="1")
ax.plot(np.random.standard_normal(1000).cumsum(), color="red", label="2")
ax.plot(np.random.standard_normal(1000).cumsum(), color="green", label="_3_")
ax.set_xlim()  # plot range
ax.set_xticks([0, 250, 500, 750, 1000])  # plot tick locations
ax.set_xticklabels(
    ["one", "two", "three", "four", "five"], rotation=30, fontsize=8
)  # tick labels
ax.set_xlabel("Stages")
ax.set_title("I am the god of graphs")
# we can use set method to set multiple things at ones
ax.set(title="My first matplotlib plot", xlabel="Stages")
# To exclude one or more elements from the legend, pass no label or label="_nolegend_"
ax.legend(loc="best")

# to draw text on a specific location on the plot, we can also draw .arrow and .annotate
ax.text(120, 10, "Hello world!", family="monospace", fontsize=10)

# exaple of adding annotations with arrows
from datetime import datetime

crisis_data = [
    (datetime(2007, 10, 11), "Peak of bull market"),
    (datetime(2008, 3, 12), "Bear Stearns Fails"),
    (datetime(2008, 9, 15), "Lehman Bankruptcy"),
]
for date, label in crisis_data:
    ax.annotate(
        label,
        xy=(date, spx.asof(date) + 75),
        xytext=(date, spx.asof(date) + 225),
        arrowprops=dict(facecolor="black", headwidth=4, width=2, headlength=4),
        horizontalalignment="left",
        verticalalignment="top",
    )

# Zoom in on 2007-2010
ax.set_xlim(["1/1/2007", "1/1/2011"])
ax.set_ylim([600, 1800])

ax.set_title("Important dates in the 2008–2009 financial crisis")

# Some of these, like Rectangle and Circle, are found in matplotlib.pyplot, but the full set is located in matplotlib.patches
# And we can add this figures to charts
fig, ax = plt.subplots()

rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color="black", alpha=0.3)
circ = plt.Circle((0.7, 0.2), 0.15, color="blue", alpha=0.3)
pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color="green", alpha=0.5)

ax.add_patch(rect)
ax.add_patch(circ)
ax.add_patch(pgon)

# Saving to files
fig.savefig("figpath.svg")

# to eddit global plot parametrs use rc obj
plt.rc("figure", figsize=(10, 10))
plt.rcParams  # config could be found here
plt.rcdefaults()  # resets config to default
# firstly parametr we want to config then configuration
plt.rc("font", family="monospace", weight="bold", size=8)

# visualisation can be done directly from pandas x is index, y is values
# plot(kind) Can be "area", "bar", "barh", "density", "hist", "kde", "line", or "pie"; defaults to "line"
s = pd.Series(np.random.standard_normal(10).cumsum(), index=np.arange(0, 100, 10))
s.plot()
# DataFrame’s plot method plots each of its columns as a different line on the same subplot, creating a legend automatically
df = pd.DataFrame(
    np.random.standard_normal((10, 4)).cumsum(0),
    columns=["A", "B", "C", "D"],
    index=np.arange(0, 100, 10),
)

# to set color schema
plt.style.use("grayscale")
df.plot()

# BAR PLOTS
fig, axes = plt.subplots(2, 1)
data = pd.Series(np.random.uniform(size=16), index=list("abcdefghijklmnop"))
data.plot.bar(ax=axes[0], color="black", alpha=0.7)
data.plot.barh(ax=axes[1], color="black", alpha=0.7)

# With a DataFrame, bar plots group the values in each row in bars, side by side, for each value
df = pd.DataFrame(
    np.random.uniform(size=(6, 4)),
    index=["one", "two", "three", "four", "five", "six"],
    columns=pd.Index(["A", "B", "C", "D"], name="Genus"),
)
df.plot.bar()
# stacked=True creates stacked bar plots
df.plot.barh(stacked=True, alpha=0.5)

# pd.crosstab() is used to create frequency table from 2 columns 1 -rows, 2nd - columns
party_counts = pd.crosstab(tips["day"], tips["size"])

# Normalize to sum rows to 1
party_pcts = party_counts.div(party_counts.sum(axis="columns"), axis="index")

# if data needs aggregation its ezier to use seaborn


# data - data table, x and y - column names, data will be aggregated by y and averaged
sns.barplot(x="tip_pct", y="day", data=tips, orient="h")
# seaborn.barplot has a hue option that enables us to split by an additional categorical value
sns.barplot(x="tip_pct", y="day", hue="time", data=tips, orient="h")
# we can change stiles with
sns.set_style("whitegrid")
# for black and wight distribution
sns.set_palette("Greys_r")
tips["tip_pct"].plot.hist(bins=50)


# Generates distribution graph wich could have generated data
data.plot.density()

comp1 = np.random.standard_normal(200)
comp2 = 10 + 2 * np.random.standard_normal(200)
values = pd.Series(np.concatenate([comp1, comp2]))
sns.histplot(values, bins=100, color="black")

ax = sns.regplot(x="m1", y="unemp", data=trans_data)
sns.pairplot(trans_data, diag_kind="kde", plot_kws={"alpha": 0.2})

# Catplots
sns.catplot(
    x="day",
    y="tip_pct",
    row="time",
    col="smoker",
    kind="bar",
    data=tips[tips.tip_pct < 1],
)
sns.catplot(x="tip_pct", y="day", kind="box", data=tips[tips.tip_pct < 0.5])
