import math
import os  # Нужно разобраться со стандартной библиотекой
import sys
import time

import numpy as np

# После переменной в интерактивном воркбуке можно поставить вопрос, чтобы получить более подробную информацию о ней
# Метод .format(var1, var2) на стринг вставит

#
#
#
# NUMPY
#
#
#


ones_matrix = np.ones((1, 5))
print(ones_matrix)
ones_submatrix_view = ones_matrix[::2, ::2]  # creates a view, not copy
ones_matrix[::2, ::2] = np.zeros((3, 3))
ones_submatrix_view

# Создание массива
data = np.array([[1.5, -0.1, 3], [0, -3, 6.5]])
# Операции без цикла
data = data * 10

# Метаданные (тупл с размером и тип данных)
data.shape
data.dtype
data.ndim

# Создание хелпер матриц для расчётов
np.zeros((3, 6))
np.empty((2, 3, 2))
np.arange(15)
np.asarray(data)
np.ones((1, 2))
np.full((10, 3), np.float16)
np.identity(4)


# Можно задавать тип через параметр dtype
arr1 = np.array([1, 2, 3], dtype=np.float64)
arr1.astype(
    np.int32
)  # Через этот метод можно конвертировать тип, всегда создаёт новый массив

# Массивы одинаковой размерности можно также сравнивать с помощью знаков
# В отличие от слайсов в питоне, слайсы в нампае не создают новый объект - это ссылка на старый
arr2 = np.ones((4, 3))
arr_slice = arr2[1]

# Для изменения всех значений в массиве используем общий слайс
arr_slice[:] = 65
arr_slice
arr2

# Если нужно скопировать вызываем метод copy
arr3 = arr2.copy()

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arr2d[2, 2]

arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[1, 0]
# Слайс н-мерных массивов проходит по самому большому пространству и работает стандартно
# Через запятые можно слайсить пространства пониже

# Индексировать строки можно также через массивы True False
# Но это всегда создаёт именно копию, если идёт присваивание к новой переменной
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
dataX = np.array([[4, 7], [0, 2], [-5, 6], [0, 0], [1, 2], [-12, -4], [3, 4]])
dataX[names == "Bob", 1:]  # Выдаст строки где True и вернёт 2:1 второй столбик

# Знак перед cond аналог !=
data[~cond]
# Можно так
mask = (names == "Bob") | (names == "Will")
data[data < 0] = 0

arr4 = np.zeros((8, 4))
for i in range(8):
    arr4[i] = i

# Для того чтобы выбрать конкретные строки передаём массив с номерами строк в нужном порядке
arr4[[2, 4, 6]]

# Также можно передавать 2 массива с индексами
arr5 = np.arange(32).reshape((8, 4))
arr5[[1, 5, 7, 2], [0, 3, 1, 2]]  # Первый выберет строки, второй - элементы в них
# fancy indexing всегда копирует при присваивании в отличие от слайсов

# Можно также задавать порядок колонок через второй слайс
arr5[[1, 5, 7, 2]][:, [0, 3, 1]]

# Транспонирование (именно вью, не копия)
arr5.T

# Dot product
dot_pr = np.dot(arr5.T, arr5)
dot_pr = arr5.T @ arr5  # Умножать также можно через собаку

# Создание массива по нормальному распределению
samples = np.random.standard_normal(size=(4, 4))
rng = np.random.default_rng(seed=12345)  # манипуляция начальным сидом
dataU = rng.standard_normal((7, 4))
dataG = rng.standard_normal((4, 4))


np.sqrt(samples)
np.exp(samples)  # считает e в степени элемента для каждого элемента
np.add(dataU, dataG)
np.max(dataU, dataG)
remainder, whole_part = np.modf(dataU)  # аналог в математике возвращает остаток и целую
out = np.zeros_like(dataU)
np.add(
    dataU, 1, out=out
)  # результат можно помещать в уже существующий массив с помощью out

points = np.arange(1, 4)
points2 = np.arange(4, 7)
xs, ys = np.meshgrid(
    points, points2
)  # Выдаст 2 матрицы, столько строк, какая длина, и сами строки - это сами массивы
# Вторая матрица - строки - сколько длина и каждая строка полностью из одного элемента второго инпута
# Далее с такими матрицами можно оперировать. По строкам первый вектор умноженный на элемент второго

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
yarr = np.array([2.1, 2.2, 2.3, 2.4, 2.5])
cond = np.array([True, False, True, True, False])
result = np.where(
    cond, xarr, yarr
)  # Элемент из первого если тру и из второго если фолс
np.where(
    data > 0, 2, -2
)  # В матрице условий на место тру поставил 2 и -2 на место фолс
dataU.mean(axis=1)  # Строка
dataU.mean(axis=0)  # Столбик
dataU.cumsum()  # Возвращает массив с промежуточными значениями на каждой итерации
# на 2д возвращается 2д итог с соответствующей суммой
arr = rng.standard_normal(100)
# True конвертится в 1 фалс в 0

ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
ints.sort()  # Сортирует сам массив а np.sort(ints) возвращает копию
np.unique(ints)  # Возвращает уже отсортированный массив уникальных значений

# Сохраняет массивы на диске: просто, в архиве без сжатия, в архиве с сжатием
np.save("./ints", ints)
np.load("ints.npy")
np.savez("array_arch.npz", a=ints, b=ints)
np.savez_compressed()

x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = np.array([[6.0, 23.0], [-1, 7], [8, 9]])
x.dot(y)  # Считаем скалярное произведение

from numpy.linalg import inv, qr

(
    np.abs(walk) >= 10
).argmax()  # argmax возвращает индекс самого большого элемента и в бинарном массиве - это тру, то есть вернёт
# индекс первого True


#
#
#
# Pandas
#
#
#

import pandas as pd

# Series - одномерный, один тип и дата лейблы доступ к индексам и самим значениям идёт через параметры
obj = pd.Series([4, 7, -5, 3])
obj.array
obj.index

obj2 = pd.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
# Индексы сохраняются при операции с Серией (фильтрация по условию, умножение и мат. операции)
# Можно также создавать из словарей

sdata = {"Ohio": 35000, "Texas": 71000, "Oregon": 16000, "Utah": 5000}
obj3 = pd.Series(sdata)
obj3.to_dict()  # Переведёт обратно в словарь

# C помощью параметра index моддно задать порядок (по дефолту порядок как в словаре)
states = ["California", "Ohio", "Oregon", "Texas"]
obj4 = pd.Series(sdata, index=states)
obj4.isna()
obj4.notna()
pd.isna(obj4)
pd.notna(obj4)

# Индексам и параметрам можно задать имена
obj4.name = "population"
obj4.index.name = "state"

# У уже созданной серии можно менять индексы
obj.index = ["Bob", "Steve", "Jeff", "Ryan"]

# Data frame named collection of columns (each column contains data of similar type, but data type can be different across the columns)
data = {
    "state": ["Ohio", "Ohio", "Ohio", "Nevada", "Nevada", "Nevada"],
    "year": [2000, 2001, 2002, 2001, 2002, 2003],
    "pop": [1.5, 1.7, 3.6, 2.4, 2.9, 3.2],
}
frame = pd.DataFrame(data)
frame.head()
frame.tail()

# Порядок колонок можно задавать через параметр columns
pd.DataFrame(data, columns=["year", "state", "pop"])
frame.columns
frame["year"]

# Строки можно получать через .loc and .iloc
frame.loc[1]

# Если словарь словарей, то первые ключи - названия колонок, вторые ключи - названия строк
populations = {
    "Ohio": {2000: 1.5, 2001: 1.7, 2002: 3.6},
    "Nevada": {2001: 2.4, 2002: 2.9},
}
frame3 = pd.DataFrame(populations)
frame3.T  # Транспонирование как в нампае и оно не сохраняет типы, если они разные вдоль колонок
frame3.columns.name = "state"
frame3.index.name = "year"
frame3.to_numpy()

labels = pd.Index(
    np.arange(3)
)  # Индексы можно создать отдельным классом и использовать в разных местах
# Значения индексов могут повторяться

obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=["d", "b", "a", "c"])
obj2 = obj.reindex(
    ["a", "b", "c", "d", "e"]
)  # Упорядочит по заданным индексам Nan если нет

obj3 = pd.Series(["blue", "purple", "yellow"], index=[0, 2, 4])
obj3.reindex(
    np.arange(6), method="ffill"
)  # Заполнит пустые ячейки предыдущими значениями
# The columns can be reindexed with the columns keyword
frame.reindex(states, axis="columns")

# Reindexing using loc
frame.loc[["a", "d", "c"], ["California", "Texas"]]

# Returns new array without specified indexes
new_obj = obj.drop("c")
data.drop(index=["two"])  # to drop rows
data.drop(columns=["two"])  # to drop columns
# use loc to find value by index bc standart way is dependable on data type
# use iloc to index specificaly with integers
# in loc and iloc slising last values are inclusive (not included in standart python)

data = pd.DataFrame(
    np.arange(16).reshape((4, 4)),
    index=["Ohio", "Colorado", "Utah", "New York"],
    columns=["one", "two", "three", "four"],
)
data.loc[
    "Colorado", ["two", "three"]
]  # to select data by both rows and columns pass them to loc separated by comma
# firstly rows then columns
# indexing is always label-oriented if index values are integers
# chain selectors return copy, sole loc returns view
# while adding 2 series indexes are merged and values are added, when value is absend at least in 1 column sum value will be NaN

# Add method to sum, and fill_value to pass to NaN
df1.add(df2, fill_value=0)
# each method has reversed counterpart starting with r, like 15.div(5) and 5.rdiv(15)
# stadart sum of series and dataframe is adding series to every row of the dataframe
# to sum columns we should add axis
frame.sub(series3, axis="index")

# numpy operations also work with pandas dataframes
np.abs(frame)


def f1(x):
    return x.max() - x.min()


# applies a function to every column of the dataframe
frame.apply(f1)
# applies to rows
frame.apply(f1, axis="columns")


# this function may also return Series
def f2(x):
    return pd.Series([x.min(), x.max()], index=["min", "max"])


frame.apply(f2)


# map applies function element wise in Series
# applymap applies function element wise in DataFrame
obj.sort_index()  # sorts indexes, with axis = 'columns' sorts column names
# default is ascending order ascending=False will use decrising order
obj.sort_values()  # used to sort by values missing are sorted to the end of the dataframe by default but with na_position="first" parametr will be forted to the begining

# we can sort dataframe by multiple columns more important columns to sort should come first
frame.sort_values(["a", "b"])
obj.rank()  # returns index at sorted array from 1 to number of elements in ascending order (if elements are equal they get similar 0.5 index)
obj.rank(
    method="first"
)  # with first method there wont be any 0.5 values using row indexes
data.rank(axis="columns", method="min")
obj.index.is_unique  # to check if indexes are unique

# if we have similar indexes the returned value will be series
df = pd.DataFrame(
    [[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]],
    index=["a", "b", "c", "d"],
    columns=["one", "two"],
)
df.sum()  # returns sums for columns DEFAULT = ACROSS THE ROWS (for column), axis columns = ACROSS THE COLUMNS = for rows
df.sum(
    axis="index", skipna=False
)  # Default - if all NA - 0, if 1 NA - its skiped, but with skapna = False its NA, mean needs at least 1 nonNA value

df.describe()  # multiple statistics for 1 shot

"""
The corr method of Series computes the correlation of the overlapping, non-NA, 
aligned-by-index values in two Series. Relatedly, cov computes the covariance:
"""
returns["MSFT"].corr(returns["IBM"])

""""
DataFrame’s corr and cov methods, on the other hand, 
return a full correlation or covariance matrix as a DataFrame, respectively
Using DataFrame’s corrwith method, you can compute pair-wise correlations between a DataFrame’s columns or rows with another Series or DataFrame.
Passing a Series returns a Series with the correlation value computed for each column
"""

# data.apply(pd.value_counts).fillna(0)

#
#
#
# CHAPTER 6 Data Loading
#
#
#
#

# Use names to name columns
pd.read_csv("examples/ex2.csv", header=None, names=names, index_col="message")
# by default first row will be considered as names of columns

# hierarchical index
parsed = pd.read_csv("examples/csv_mindex.csv", index_col=["key1", "key2"])

# If data is separated by nultiple amount of spaces
# if there is more columns than provided in names first column is considered an index
result = pd.read_csv("examples/ex3.txt", sep="\s+")

# skiprows to skip rows)
pd.read_csv("examples/ex4.csv", skiprows=[0, 2, 3])

# NaN - NA, NULL, empty string by default. We can delete default list of NA values with keep_default_na=False parametr
# The na_values option accepts a sequence of strings to add to the default list of strings recognized as missing:
result = pd.read_csv("examples/ex5.csv", na_values=["NULL"])

# Different NA sentinels can be specified for each column in a dictionary
sentinels = {"message": ["foo", "NA"], "something": ["two"]}
pd.read_csv("examples/ex5.csv", na_values=sentinels, keep_default_na=False)

# to limit rows shown
pd.options.display.max_rows = 10
pd.read_csv("examples/ex6.csv", nrows=5)

# to read file in pieces
chunker = pd.read_csv("examples/ex6.csv", chunksize=1000)
for piece in chunker:
    tot = tot.add(piece["key"].value_counts(), fill_value=0)
tot = tot.sort_values(ascending=False)

# to convert back to original type
data.to_csv("examples/out.csv", na_rep="NULL")

# With no other options specified, both the row and column labels are written. Both of these can be disabled
data.to_csv(sys.stdout, index=False, header=False)

# You can also write only a subset of the columns, and in an order of your choosing
data.to_csv(sys.stdout, index=False, columns=["a", "b", "c"])

# To read csv with standart python
import csv

f = open("examples/ex7.csv")
reader = csv.reader(f)
for line in reader:
    print(line)
f.close()

with open("examples/ex7.csv") as f:
    lines = list(csv.reader(f))
header, values = lines[0], lines[1:]

data_dict = {h: v for h, v in zip(header, zip(*values))}


# We can specify delimeter and other parsing options
class my_dialect(csv.Dialect):
    lineterminator = "\n"
    delimiter = ";"
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL


reader = csv.reader(f, dialect=my_dialect)
reader = csv.reader(f, delimiter="|")

# to write to csv with standart python
with open("mydata.csv", "w") as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(("one", "two", "three"))
    writer.writerow(("1", "2", "3"))
    writer.writerow(("4", "5", "6"))
    writer.writerow(("7", "8", "9"))


import json

result = json.loads(obj)  # to read json
asjson = json.dumps(result)  # to convert object to json

# The default options for pandas.read_json assume that each object in the JSON array is a row in the table
data = pd.read_json("examples/example.json")


# XML parsing
from lxml import objectify

path = "datasets/mta_perf/Performance_MNR.xml"
with open(path) as f:
    parsed = objectify.parse(f)
root = parsed.getroot()


data = []

skip_fields = ["PARENT_SEQ", "INDICATOR_SEQ", "DESIRED_CHANGE", "DECIMAL_PLACES"]

for elt in root.INDICATOR:
    el_data = {}
    for child in elt.getchildren():
        if child.tag in skip_fields:
            continue
        el_data[child.tag] = child.pyval
    data.append(el_data)
perf = pd.DataFrame(data)

# Bytes to disk
frame.to_pickle("examples/frame_pickle")
pd.read_pickle("examples/frame_pickle")

# conda install pyarrow to read parquet files
fec = pd.read_parquet("datasets/fec/fec.parquet")

# Excel files
# conda install openpyxl xlrd
xlsx = pd.ExcelFile("examples/ex1.xlsx")
xlsx.sheet_names
xlsx.parse(sheet_name="Sheet1")
xlsx.parse(sheet_name="Sheet1", index_col=0)

# write to excel
writer = pd.ExcelWriter("examples/ex2.xlsx")
frame.to_excel(writer, "Sheet1")
writer.close()
# or simply
frame.to_excel("examples/ex2.xlsx")

#
#
#
# HDF5
#
#
#

# conda install pytables
frame = pd.DataFrame({"a": np.random.standard_normal(100)})
store = pd.HDFStore("examples/mydata.h5")
store["obj1"] = frame
store["obj1_col"] = frame["a"]
store.put("obj2", frame, format="table")
store.select("obj2", where=["index >= 10 and index <= 15"])
# read hdf with pandas
pd.read_hdf("examples/mydata.h5", "obj3", where=["index < 5"])


# conda install requests
import requests

resp = requests.get(url)
resp.raise_for_status()
data = resp.json()


# from sql request
# conda install sqlalchemy
import sqlalchemy as sqla

db = sqla.create_engine("sqlite:///mydata.sqlite")
pd.read_sql("SELECT * FROM test", db)
