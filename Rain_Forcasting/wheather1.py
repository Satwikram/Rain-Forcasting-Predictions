# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:48:16 2020

@author: SATWIK RAM K
"""
#Importing the libraries
import numpy as np
import pandas as pd

#Importing dataset
dataset = pd.read_csv('seattleWeather_1948-2017.csv')

#In the dataset we don't require dates as we can't predict the rain using date

#Getting dummies
df = pd.get_dummies(dataset['RAIN'], drop_first = True)

#Concat dummies
dataset = pd.concat([dataset, df], axis = 1)

#Dropping original values
dataset.drop('RAIN', axis = 1, inplace = True)

#Splitting the data to x and y
x = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

#Checking for nan values 
inds = np.where(np.isnan(x))
indss = np.where(np.isnan(y)) #Empty Array

#Yes nan exist, we have to fill nan values with mean
#Place column means in the indices. Align the arrays using take
col_mean = np.nanmean(x, axis = 0)
x[inds] = np.take(col_mean, inds[1])

#Check again for nan value exist
inds1 = np.where(np.isnan(x)) # getting empty array

#Splitting data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42 )

#Feature scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Importing Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(x_train, y_train)

# Predicting the Test set results
y_pred = model.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) #97% accuracy

#Importing Decision Tree 
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred1 = classifier.predict(x_test)

# Making the Confusion Matrix
cm1 = confusion_matrix(y_test, y_pred1) #100% accuracy

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier1.fit(x_train, y_train)

# Predicting the Test set results
y_pred2 = classifier.predict(x_test)

# Making the Confusion Matrix
cm2 = confusion_matrix(y_test, y_pred2)



