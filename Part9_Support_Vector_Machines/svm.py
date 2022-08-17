#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:57:20 2022

@author: aman
"""
import pandas as pd

dataset = pd.read_csv("01Exercise1.csv")
loan_prep = dataset.copy()

# Dropping irrelevent column of "gender". We are not dropping "married" because
# it is considered that married people have more stability so it can have impact.
loan_prep = loan_prep.drop(['gender'], axis=1)

# Identify the missing values
loan_prep.isnull().sum()

# Dropping all rows with missing values :
loan_prep = loan_prep.dropna()

# Now again checking missing values
loan_prep.isnull().sum()

# Creating dummy variables for categorical columns :
print(loan_prep.dtypes);

# Creating dummy variables :
loan_prep = pd.get_dummies(loan_prep, drop_first=True)

# Normalize data for loanamt and income using StandardScalar
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

loan_prep['income'] = scalar.fit_transform(loan_prep[['income']])
loan_prep['loanamt'] = scalar.fit_transform(loan_prep[['loanamt']])

# Splitting dataset into test set and train set
X = loan_prep.iloc[: , :-1].values;
y = loan_prep.iloc[: , -1].values;

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=1234, stratify=y)

#-------------------------------------------------------------
#-------------------------------------------------------------
# Till here everything was same as logistic_regression.py file
#-------------------------------------------------------------
#-------------------------------------------------------------

# Build the logistic regression model :
from sklearn.svm import SVC
svc = SVC();
svc.fit(X_train, y_train);

y_predict = svc.predict(X_test)


# Calculating score
score = svc.score(X_test, y_test)    

# Build the confusion matrix to compare result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)

print(cm)
print(score)
"""
In case of simply dropping all missing value rows 
score => 0.791, cm =>
------------------------
[[ 20  31]
 [  3 109]]
------------------------
"""
