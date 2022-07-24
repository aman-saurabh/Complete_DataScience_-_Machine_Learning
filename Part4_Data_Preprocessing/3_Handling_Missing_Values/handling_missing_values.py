#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handling Missing Values
Created on Mon Jul 25 00:24:08 2022

@author: aman
"""

# Importing required libraries
import pandas as pd;

# Reading the CSV file
dataset = pd.read_csv("loan_small.csv");

# Find the number of missing values for each column
dataset.isnull().sum(axis=0)
"""
It will print output as follows : 
Using this data you can decide if you should replace missing values with 
suitable replacement or you should simply drop the column (preferred if 
there are many missing values for any column)
----------------------------------------------------
Loan_ID              0
Gender               1
ApplicantIncome      2
CoapplicantIncome    1
LoanAmount           3
Area                 1
Loan_Status          1
----------------------------------------------------
"""
# Part1 :- Dropping missing values.
# To drop all data(i.e rows) with missing values(i.e missing value for any column)
cleandata = dataset.dropna();
"""
Now if you see the cleandata by clicking on 'cleandata' in variable explorer 
You will find in this dataframe there is no missing data(represented by 'nan')
for any column. 
But this should not be preferred method as in this method we can loose many 
data like here also we loosed 7 data out of 16 i.e almost half of the data.
So instead we should drop a particular column or only those rows which have 
missing data for a particular column(especially for column of dependent 
variable i.e which data which we want to predict using our model)
"""

# To drop all data(i.e rows) with missing values for some particular columns
cleandataC = dataset.dropna(subset=['Loan_Status']);
"""
Now you can see we have dropped only 1 row of data and still we have majority 
of data. 
"""

# To drop all data(i.e rows) with atleast 2 missing values 
cleandataCN = dataset.dropna(thresh=2);

# To drop all columns with missing data(in any row).
cleandataN = dataset.dropna(axis=1)
#Or
cleandataN = dataset.dropna(axis='columns')

# To drop columns with 2 or more missing values.
cleandataNN = dataset.dropna(axis=1, thresh=(len(dataset.index) - 2))
#Or
cleandataNN = dataset.dropna(axis='columns', thresh=(dataset.shape[0] - 2))
"""
Explanation :-
We don't have any direct method to drop a column with this condition but we 
have a method using which we can define - keep all columns which have atleast
defined number of non-missing values. For example we want to keep all colums 
that have 2 or more non-missing values and drop rest. We can achieve as 
follows :
"""
#cleandataNN = dataset.dropna(axis='columns', thresh=2)
"""
Using it We can also drop all columns that have more than given number of 
missing data. But for that we will need total number of rown in the dataset 
which we can get using following methods :
len(dataset.index)
dataset.shape[0]
Now we can drop columns with 3 or more missing values as follows
"""
#cleandataNN = dataset.dropna(axis=1, thresh=(len(dataset.index) - 2))
#Or
#cleandataNN = dataset.dropna(axis='columns', thresh=(dataset.shape[0] - 2))


# Few more important concepts regarding dropping columns and rows
# Drop columns by name
cleandataDC = dataset.drop(['Gender', 'Loan_ID'], axis=1)
# Or
cleandataDC = dataset.drop(columns=['Gender', 'Loan_ID'])

# Drop rows by index
"""
To drop second and fifth rows(i.e. index 1 and 4)
"""
cleandataDR = dataset.drop([1, 4])

# Part2 :- Replacing categorical missing values :
# First copying the datasetinto a new variables :
dt = dataset.copy();

# Replacing categorical values with mode :
"""
In the dataset "Gender", "Area" and "Loan_Status" are categorical data
(i.e value is in string or boolean format). For categorical data missing values 
should be replaced using "mode" method which basically means most frequest 
values. But in a dataframe there can be multiple values mode(for example - 
in the area column if the number of 'urban' and 'semi' is same and highest then 
both will be mode.), So mode always gives value in the array format. So we 
should use it's anyone value as replacement. Best option is to use it's first 
value as we don't know how many values mode will return but we know it will 
return atleast 1 value. We can get the first value using its ".iloc[0]" 
property. So in the following syntax ".iloc[0]" is used for selecting first 
value of mode.
"""
# Storing names of categorical type columns in a variables :
cat_cols = ['Gender', 'Area', 'Loan_Status']

# Replacing categorical missing values with mode :
dt[cat_cols] = dt[cat_cols].fillna(dt.mode().iloc[0])
"""
Now if you see number of missing values for each column, you will find there is 
no missing values for 'Gender', 'Area' and 'Loan_Status' as we have replaced 
their missing values with their mode (i.e. with their most frequent values).
But note that we have replaced values in "dt" and not in "dataset" itself. So 
check it for "dt" and not for "dataset"
"""
# Find the number of missing values for each column of "dt".
dt.isnull().sum(axis=0)


# Part3 :-Replacing numerical missing values :
# Storing names of numerical type columns in a variables :
num_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

# Replacing numerical missing values with mean :
dt[num_cols] = dt[num_cols].fillna(dt.mean(numeric_only=True))
"""
If you don't pass "numeric_only=True" as argument in it you will get following 
error :
FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 
'numeric_only=None') is deprecated; in a future version this will raise 
TypeError.  Select only valid columns before calling the reduction.

It is because there are some columns(like 'Gender', 'Area' and 'Loan_Status') 
that contain strings, but df.mean() only works with columns that contain 
numbers (floats, ints, nan, etc.). So by "numeric_only=True" we are saying 
ignore columns that are non-numeric only calculate the mean for columns that 
only contain numbers.

You might be thinking here we are replacing values for 'ApplicantIncome', 
'CoapplicantIncome' and 'LoanAmount' columns only then what it has to do with
'Gender', 'Area' and 'Loan_Status' columns. Here if you notice you will see we
have applied "dt.mean()" so it will try to find mean for all columns. However 
since we have applied "fillna" function only for "num_cols" so changes will be 
reflected only on those columns which are part of "num_cols"
"""


"""
Now if you see number of missing values for each column, you will find there is 
no missing values for 'Gender', 'Area' and 'Loan_Status' as we have replaced 
their missing values with their mode (i.e. with their most frequent values).
But note that we have replaced values in "dt" and not in "dataset" itself. So 
check it for "dt" and not for "dataset"
"""
# Find the number of missing values for each column of "dt".
dt.isnull().sum(axis=0)


    