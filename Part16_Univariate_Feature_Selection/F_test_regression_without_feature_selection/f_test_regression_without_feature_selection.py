# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 02:18:47 2022

@author: Aman Saurabh
"""

# Import libraries
import pandas as pd

# Read the file
f = pd.read_csv("Students2.csv")

# Split the columns into Depndent(Y) and Independent(X) features
X = f.iloc[:, :-1]
y = f.iloc[:, -1]

# Perform Linear regression using original dataset
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

# Split the data into train set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.4, random_state=1234)
    
# Fit the data and predict Y
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)

# Calculate the RMSE error for the regression
from sklearn.metrics import mean_squared_error
import math

rmse = math.sqrt(mean_squared_error(y_test, y_predict))

##################################
### Steps for feature analysis ###
##################################

# Import f_regression from sklearn
from sklearn.feature_selection import f_regression as fr
result = fr(X,y)

f_score = result[0]
p_values = result[1]

# Print the table of features, f-score and p-values
columns = list(X.columns)
print(" ")
print(" ")
print(" ")

print("    Features    ", "    F-scores    ", "    P-values    ")
print("  ------------  ", "  ------------  ", "  ------------  ")

for i in range(0, len(columns)):
    f1 = "%4.2f" % f_score[i]
    p1 = "%2.6f" % p_values[i]
    print("   ", columns[i].ljust(12), " ", f1.rjust(8),"         ", p1.rjust(8))


"""
So as we can see only features "Hours" and "sHours" have P-values less than the
threshold value of 0.05 and other features P-values are way higher. So we will
select these 2 features only to build our more accurate model. So let's build 
our next model and check if there is any improvement in performance or not. 
"""

# Perform the Linear Regression with reduced features
X_train_n = X_train[['Hours', 'sHours']]
X_test_n = X_test[['Hours', 'sHours']]

lr1 = LinearRegression()
lr1.fit(X_train_n, y_train)

y_predict_n = lr1.predict(X_test_n)

# Calculate the RMSE with reduced features
rmse_n = math.sqrt(mean_squared_error(y_test, y_predict_n))
"""
So as we can see the RMSE error now is reduced to approx. 5 from approx. 7 
earlier. So from here we can see that how dropping the irrelevant features 
helps in improving the models performance. Apart from that such models will 
take less time in training also as it has now less features. 
"""

# ********************Important************************
"""
You can use Chi-square also in the similar manner for categorical dataset
ScikitLearn function name for Chi-square is "chi2" and you can import it as 
follows :
from sklearn.feature_selection import chi2 
"""