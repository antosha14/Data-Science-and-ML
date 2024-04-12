import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

dataset = pd.read_csv("./Bitcoin Historical Data.csv")

# iloc использует сначала ряды, потом колонки
# values возвращает массив
X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values


# Handling missing data
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X)
X = imputer.transform(X)


# Encoding categorical data: one_hot_encoding - создаст для каждого элемента таргетной колонки, 3 колонки с 1,0 векторами
# Второй параметр отвечает за то оставлять ли колонки, к которым не применяется трансформация или нет
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)
# Методы тренировки ожидают именно нампай массив, поэтому насильно конвертируем через np.array
X = np.array(ct.fit_transform(X))

# Энкодит одну колонку в зависимости от числа уникальных элементов
le = LabelEncoder()
y = le.fit_transform(y)

# Разбивка на тренировочный и тестовый набор
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# Стандартизация
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Скейлить также нужно и фичи новых данных, но делать это нужно используя ранее созданный скейлер (то есть только метод transform)
X_test = sc.transform(X_test)
