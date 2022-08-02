#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One hot encoding 
Created on Tue Jul 26 22:04:32 2022

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


# Label encoding
# Changing the datatype of categorical columns from "object" to category
dt[cat_cols] = dt[cat_cols].astype('category')

# Replacing categorical values with corresponding numerical values
for col in cat_cols:
    dt[col] = dt[col].cat.codes
    
# One hot encoding
# Dropping column - Loan_ID as it doesn't make impact on loan status.
"""
It isn't part of one hot encoding but it's a part of data cleaning and is a 
necessary step as columns like Loan_ID have no impact the dependent variable 
(i.e. Loan_Status in this case) but can create unnecessary hurdle in processing 
our data. Even it will create issue in creating one hot encoding also. So 
removing it from the dataset.
If we comment out or remove "label encoding" steps then we can use "dt" also
inplace of dataset in the following example.
"""
dt1 = dataset.drop(['Loan_ID'], axis=1) 
"""
Note that here we have used "dataset" and not "dt". It's because for label 
encoding we had converted datatype of categorical columns from "object" to 
"category" but "get_dummies()" method of Pandas which we are using below for 
one hot encoding, create dummies only for "object" type data. So if we use "dt" 
instead of "dataset" then you will find no impact of "get_dummies" method as in 
"dt" we had applied label encoding and changed the datatype of categorical 
columns from "object" to "category". 
Actually the datatype of "LoanID" was also "object", so if we don't dropped
this column then get_dummies() method will create one dummy column for each 
LoanID i.e we will get many unnecessary columns, so that was the main reason 
why we dropped "LoanId" column before performing one hot encoding.
"""

# Applying one hot encoding i.e. creating dummy variables(i.e columns) for 
# each type of categorical data
dt2 = pd.get_dummies(dt1)
"""
Now you will see see the number of columns have increase from 6 to 10. It 
because instead of 1 column for "gender" now we have 2 columns for it. One 
representing "Male" and another representing "Female". Similarly instead of
1 column for "Area" now we have 3 columns and instead of 1 column for 
"Loan_Status" now we have 2 for it. So there is an increase of 
(2-1)+(3-1)+(2-1) = 4 columns.
"""
"""
In several cases like in "Multiple Linear Regression" we should create 1 less 
column from the number of values. Since the assumptions of Multiple Linear 
Regression says that there shouldn’t be a correlation between independent 
variables. If we keep all dummy variables of the set of dummy variables then 
there will be a correlation between these dummy variables. It is commonly 
known as “Dummy variable trap”. So in case of dummy variables, we should dump 
one dummy variable from each set and consider only rest. 
For achieving this(i.e for creating 1 less dummy variable from each set) we 
just need to pass "drop_first= True" as parameter in the "get_dummies" method.
"""
dt3 = pd.get_dummies(dt1, drop_first=True)
"""
Now you will see there are 7 columns only instead of 10. It's because it 
dropped columns for "Gender_Female", "Area_Rural" and "Loan_Status_N".
"""