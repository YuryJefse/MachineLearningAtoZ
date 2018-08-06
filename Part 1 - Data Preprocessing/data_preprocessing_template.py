# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('Data.csv')

#Get the Independent variables
X = dataset.iloc[:, :-1].values

#Get the dependent variable
Y = dataset.iloc[:, 3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN")
imputer = imputer.fit(X[:, 1:])
X[:,1:] = imputer.transform(X[:,1:])


#Encoding categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_country = LabelEncoder()
X[:,0] = labelencoder_country.fit_transform(X[:,0])
labelencoder_purchased = LabelEncoder()
Y = labelencoder_country.fit_transform(Y)

#Create Dummy variables
onehotencoder_country = OneHotEncoder(categorical_features = [0])
X = onehotencoder_country.fit_transform(X).toarray()

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)