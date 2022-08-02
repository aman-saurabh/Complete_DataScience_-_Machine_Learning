#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Normalization :-
Created on Tue Jul 26 22:55:46 2022

@author: aman
"""
"""
In this part we are focusing on data normalization. So we are not performing 
previous preprocessing steps to keep file simple, clean and easy to understand.
"""
# Importing required libraries
import pandas as pd;

# Reading the CSV file
dataset = pd.read_csv("loan_small.csv");

# Dropping rows with missing value
"""
This is not a standard approach but to focus on data normalization methods
we have skipped all standard steps and simply dropped rows with missing values 
"""
cleandata = dataset.dropna()

# Extracting the three numerical columns
data_to_scale = cleandata.iloc[:, 2:5]
"""
Here all the numerical columns are in continuation so we can apply above method
But suppose if they weren't in continuation. In that case we can extract such 
columns as follows :
data_to_scale = cleandata.iloc[:, [2,3,4]]
"""

# Code for data normalization of numerical columns:
# Method 1 :- Z-transformation or Standardization 
# Importing standard scalar from scikit-learn
from sklearn.preprocessing import StandardScaler
scalar_ = StandardScaler()
ss_scalar = scalar_.fit_transform(data_to_scale)

# Method 2 :- Min-Max normalization
from sklearn.preprocessing import minmax_scale
mm_scalar = minmax_scale(data_to_scale)







