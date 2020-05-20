import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.impute import SimpleImputer


# categorial data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('Data.csv')

print(dataset)
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

print('\n')


# mean strategy hehe
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)


# categorial data
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)
