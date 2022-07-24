#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 01:59:42 2022

@author: aman
"""

"""
Label encoding simply means converting categorical data into corrsponding 
numerical values. Most people achieve it using "scikit learn" but we can 
achieve it using pandas also. So let's see how can we achieve that using 
"pandas" 
"""

# Importing required libraries
import pandas as pd;

# Reading the CSV file
dataset = pd.read_csv("loan_small.csv");

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

"""
If you check the datatype for 'Gender', 'Area', 'Loan_Status' you will find it 
is "object". But for it's conversion into numerical types using pandas we need 
to convert it into "category" type. As for this we will need to use ".cat" 
accessor and .cat accessor can be used with a 'category' dtype only. 
So we can change it's datatype to 'category' as follows :  
"""
# Checking the datatype for all columns
dt.dtypes

# changing the datatype of categorical columns from "object" to category
dt[cat_cols] = dt[cat_cols].astype('category')

# Checking the datatype for all columns again
dt.dtypes

# Now we can replace the categorical values with corresponding numerical values
# using the following code.
for col in cat_cols:
    dt[col] = dt[col].cat.codes
    
"""
You might be thinking instead of running loop why don't we apply this 
".cat.codes" in the "dt[cat_cols]" itself. We can achieve that as follows :
dt[cat_cols] = dt[cat_cols].cat.codes
Actually as discussed earlier also ".cat" accessor works with ''category' 
datatype data only and we can use it for one column only at a time. So we have 
to run it one by one only and for that running it using loop is best way. 
"""

