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
y = dataset.iloc[:, -1].values

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
y = le.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# feature scaling
# standardization = x-mean(x)/standard deviation(x)  will do it all the time
# normalization = (x - min(x))/(max(x)-min(x)) for specific tasks with value of standard distribution

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
