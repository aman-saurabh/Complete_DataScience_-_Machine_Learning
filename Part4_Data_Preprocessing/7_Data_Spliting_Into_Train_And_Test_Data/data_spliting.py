#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:52:31 2022

@author: aman
"""
# Importing required libraries
import pandas as pd;

# Reading the CSV file
dataset = pd.read_csv("loan_small.csv");

# Handling missing data
# Copying the datasetinto a new variables :
dt = dataset.copy();

# Storing names of categorical type columns in a variables :
cat_cols = ['Gender', 'Area', 'Loan_Status']

# Replacing categorical missing values with mode :
dt[cat_cols] = dt[cat_cols].fillna(dt.mode().iloc[0])

# Storing names of numerical type columns in a variables :
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

# Replacing numerical missing values with mean :
dt[num_cols] = dt[num_cols].fillna(dt.mean(numeric_only=True))

# One hot encoding
# Dropping column - Loan_ID as it doesn't make impact on loan status.
dt = dt.drop(['Loan_ID'], axis=1) 
# Applying one hot encoding i.e. creating dummy variables(i.e columns) for 
# each type of categorical data dropping one such dummy column
dt = pd.get_dummies(dt, drop_first=True)

# Steps for spliting data into training data and test data
# Spliting the data into independent and dependent variable
"""
Here X represent independent variable and y represents dependent variable
"""  
X = dt.iloc[:, :-1] 
# Selecting all rows and all columns except the last one as it is dependent variable.
y = dt.iloc[:, -1]
# Selecting all rows and only the last column as it is only dependent variable.

# Spliting the data into train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, random_state=123)

