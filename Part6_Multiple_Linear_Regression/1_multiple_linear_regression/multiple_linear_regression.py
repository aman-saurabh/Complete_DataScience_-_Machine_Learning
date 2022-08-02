#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiple Linear Regression
Created on Thu Jul 28 21:52:55 2022

@author: aman
"""

# Importing required libraries
import pandas as pd;

# Reading the CSV file
dataset = pd.read_csv("02Students.csv");
dt = dataset.copy();
"""
Since there is no missing data in our dataset so skipped handling missing 
data steps. If your dataset has missing data then apply those steps also.
Normalization is mostly not required for Simple Linear Regression so skipped
that steps also. And even if required then LinearRegression class of 
sklearn.linear_model package itself can do it for us, we just have to pass 
"normalize=true" in the regressor object as parameter. So normalization steps
are not required here.
"""
# Spliting the data into independent and dependent variable
X = dt.iloc[:, :-1].values
y = dt.iloc[:, -1].values

# Spliting the data into train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=123)

# Train the multiple linear regression model
from sklearn.linear_model import LinearRegression

# Creating regressor of LinearRegression class
std_reg = LinearRegression()

# Train and fit the training data
std_reg.fit(X_train, y_train)

# Creating prediction - let's predict the values of Y from the test data
y_predict = std_reg.predict(X_test) 

# R-squared and the equation of the line
mlr_score = std_reg.score(X_test, y_test)

# Coefficient of the line(i.e b1)
mlr_coefficient = std_reg.coef_
# Intercept of the line(i.e b0)
mlr_intercept = std_reg.intercept_

# Equation of the line :- 
"""
y = -6.19 + 5.14 * Hours + 5.82 * sHours
"""

# Calculate Root mean squared error(RMSE) :-
# To know how much error our model has made
from sklearn.metrics import mean_squared_error
import math
mlr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))