# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the Dataset
dataset = pd.read_csv('/Users/yuryjefse/Documents/Estudo/MachineLearningAtoZ/Part 1 - Data Preprocessing/Data.csv')

#Get the Independent variables
X = dataset.iloc[:, :-1].values

#Get the dependent variable
Y = dataset.iloc[:, 3].values

#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN")
imputer = imputer.fit(X[:, 1:])
X[:,1:] = imputer.transform(X[:,1:])
