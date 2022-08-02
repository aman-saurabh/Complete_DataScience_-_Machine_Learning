#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple linear Regression
Created on Thu Jul 28 00:01:52 2022

@author: aman
"""
# Importing required libraries
import pandas as pd;

# Reading the CSV file
dataset = pd.read_csv("01Students.csv");
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

# Train the simple linear regression model
from sklearn.linear_model import LinearRegression
"""
Details of LinearRegression class :-
Inportant paramters:-
fir_intercept :- Default is true
True means it tells the regressor to calculate the intercept.And False means 
it tells regressor not to calculate the point of intersection and it assumes 
that regression line passes through (0,0) i.e. (0,0) is the point of 
intersection.
normalize :- Default is False
False means don't normalize the data.
----------------------------------------------------------------------
Important attributes:-
coef_ - > Coefficient of regression(i.e b1)
intercept_ -> intercept(i.e b0)
"""
# Creating regressor of LinearRegression class
std_reg = LinearRegression()

# Train and fir the training data
std_reg.fit(X_train, y_train)

# let's now predict the values of Y from the test data
y_predict = std_reg.predict(X_test) 


# R-squared and the equation of the line
slr_score = std_reg.score(X_test, y_test)
"""
The max value for R-quared is 1 which means model predicts accurately but it 
rarely returns 1. So a value near to 1 indicates that model is good in 
predicting results. Our score is 0.847 which is close to 1 and hence our model 
is quiet good.
"""

# Coefficient of the line(i.e b1)
slr_coefficient = std_reg.coef_
# Intercept of the line(i.e b0)
slr_intercept = std_reg.intercept_

# Equation of the line :- 
"""
With the help of slr_coefficient and slr_intercept we can find the equation of 
the line. As equation of the line is
y = b0 + b1 * x
Here 
b0 = slr_intercept = 33.81
and 
b1 = slr_coefficient[0] = 5.358 
//Since slr_coefficient is returned in list format. Since same LinearRegression
class is used for Multiple Linear regression also and incase of Multiple Linear 
regression there will be multiple coefficients(1 for each independent variable)
But since this is a Simple linear regression and we have only one independent 
variable, so it will return only one result in the list so we picked the first
one.
Now since,
y = b0 + b1 * x
So our Equation for the line will be :
-------------------------------
y = 33.81 + 5.358 * x
-------------------------------
"""

# Calculate Root mean squared error(RMSE) :-
# To know how much error our model has made
from sklearn.metrics import mean_squared_error
import math
slr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))


# Plotting the result using matplotlib.pyplot :-
import matplotlib.pyplot as plt

# Step11 - Visualising the training set result
y_train_predict = std_reg.predict(X_train)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, y_train_predict, color='blue')
plt.title('Study hours va Marks(Training Set)')
plt.xlabel("Study hours")
plt.ylabel("Marks")
plt.ylim(ymin=0)    #To make sure that y axis start at zero.
plt.show()

# Step12 - Visualising the test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_predict, color='blue')
plt.title('Study hours va Marks(Test Set)')
plt.xlabel("Study hours")
plt.ylabel("Marks")
plt.ylim(ymin=0)
plt.show()




