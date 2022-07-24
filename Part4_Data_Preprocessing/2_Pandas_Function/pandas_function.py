"""'
Functions of pandas

Created on Sun Jul 24 23:54:23 2022

@author: aman
"""

# Importing required libraries
import pandas as pd;

# Reading the CSV file
dataset = pd.read_csv("loan_small.csv");

# head() - To get a quick view of the dataset
"""
Note :- 
1.) It will run if you press "Ctrl + Shift + Enter"(i.e if you run 
selection). If you run file or if you run cell you will not get output in the 
terminal. For that enclose the following code inside print() statement.
2.) The default size for "head" method is 5 i.e if you don't pass any value as 
parameter you will get first 5 results. Here since we have passed 10 so in this 
case we will get first 10 results.
"""
dataset.head(10);

# shape - To get the shape of the dataset
"""
Note that here we haven't applied parenthesis at the end as it is a property 
and not a function.
"""
dataset.shape

# columns - To get the column names of the file
dataset.columns


# copy() - To copy the dataset
dt = dataset.copy()