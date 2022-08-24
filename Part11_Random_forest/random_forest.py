# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 01:05:07 2022

@author: Aman Saurabh
"""

# Importing libraries
import pandas as pd

# Read dataset
data = pd.read_csv("decisiontreeAdultIncome.csv")

# Check for null values
data.isnull().sum(axis=0)
# So there is no missing values. If we was we already know how to handle them.

# Check datatye of columns
data.dtypes
# Since datatype of all categorical columns is object so we don't need to 
# update them as pandas get_dummies() function work for object as well as 
# categorical types

# Create dummy variables
data_prep = pd.get_dummies(data, drop_first=True)
  
# Create X and Y variables 
X = data_prep.iloc[:, :-1]
y = data_prep.iloc[:, -1]

# Create train and test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=1234, stratify=y)

# Create the model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1234)
rfc.fit(X_train, y_train)
y_predict = rfc.predict(X_test)

# Evaluate the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
score = rfc.score(X_test, y_test)
# Out of 5937 records (3889+856)=4745 are correct prediction and(708+484)=1192
# are wrong prediction and the score is 0.7999. So here as we can see the 
# number of correct predictions and score is improved in comparison to 
# decision tree model









