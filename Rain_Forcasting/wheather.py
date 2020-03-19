# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 14:44:14 2020

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

#Done with data cleaning proceding to predict

#Importing Tensorflow Libraries
import tensorflow as tf
from tensorflow import keras


#Building the model
model = tf.keras.models.Sequential([
        #Adding first and hidden layer
        tf.keras.layers.Dense(input_dim = 3, units = 256, kernel_initializer = 'uniform', activation = 'relu'),
        tf.keras.layers.Dropout(0.1),
        
        #Adding Second Hiddenlayer
        tf.keras.layers.Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'),
        tf.keras.layers.Dropout(0.1),
        
        #Adding Third Hidden layer
        tf.keras.layers.Dense(units = 1024, kernel_initializer = 'uniform', activation = 'relu'),
        tf.keras.layers.Dropout(0.1),
        
        #Adding Last Output Layer
        tf.keras.layers.Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')
        
        ])

#Compiling the model
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Building the checkpoint
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('accuracy') >= 1.0):
            print("\n Reached 99% of accuracy so Stopped Training!")
            self.model.stop_training = True
        
callback = mycallback()        
        
#Fitting the model
model.fit(x_train, y_train, batch_size = 100, epochs = 100, callbacks = [callback])

#Predicting the test set
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)

#Making Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm  = confusion_matrix(y_test, y_pred)
score = accuracy_score(y_test, y_pred)

#Making Single prediction
# Enter data in order 
new_pred = model.predict(sc.transform(np.array([[0.5, 43, 50]])))
new_pred = (new_pred > 0.5)

