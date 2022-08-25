#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 02:54:07 2022

@author: aman
"""
import pandas as pd

dataset = pd.read_csv("01Exercise1.csv")
# Note in the dataset "ch" represents whether customer has credit history or not.

loan_prep = dataset.copy()

# Dropping irrelevent column of "gender". We are not dropping "married" because
# it is considered that married people have more stability so it can have impact.
loan_prep = loan_prep.drop(['gender'], axis=1)

# Identify the missing values
loan_prep.isnull().sum()
"""
Output is :
married     3
ch         50
income      0
loanamt    22
status      0

So married, ch and loanamt has missing values. 
"""
# Dropping all rows with missing values :
loan_prep = loan_prep.dropna()
"""
It will remove 73 records which is not very high for 614 records but not very 
low also so let's try to replace missing values with suitable values.
For numerical columns we will use mean and for categorical columns we will use 
mode.
Among these missing data columns loanamt is numerical column and married and ch
are categorical columns.
"""
# Replacing missing values with suitable replacements
"""
# Replacing numerical values with mean()
missing_num_col = ['loanamt']
loan_prep[missing_num_col] = \
loan_prep[missing_num_col].fillna(loan_prep.mean(numeric_only=True))

# Replacing categorical missing values with mode()
missing_cat_col = ['married', 'ch']
loan_prep[missing_cat_col] = \
loan_prep[missing_cat_col].fillna(loan_prep.mode().iloc[0])
"""

# Now again checking missing values
loan_prep.isnull().sum()
"""
Now you will find there is no missing values.
After completing and executing this program with replacing missing values 
uncomment line in which we dropped all rows with missing values i.e.
------------------------------------------
loan_prep = loan_prep.dropna(); 
------------------------------------------
and comment out all above lines in which we are replacing missing values with 
mean and mode.
And then compare the confusion matrix and score and see in which we get better 
prediction. But note that whichever among above 2 approach get better result
it doesn't mean that in general that approach gives better result always.
It will be applicable for this dataset and this program only. Depending on the 
dataset and algorithm you are using you might get better result.  
"""

# Creating dummy variables for categorical columns :
"""
Checking datatypes of columns as we can apply get_dummies method only on 
"categorical" type columns. Following code itself should print datatypes of all
columns but don't know why it was not printing in my system so applied print()
method on that.
-------------------------
loan_prep.dtypes
-------------------------
"""
print(loan_prep.dtypes);
"""
Note that "married", "ch" and "status" are categorical columns, among these
"ch" column is of type "float" so we need to convert it into "categorical" type
but note that it's value is either 0 and 1 only so if we create dummy variable
for it and drop it's first column then again we will get same value(or exactly
opposite value). So not changing its type so that pandas don't create dummy 
variable for it. And other categorical columns i.e. married and status are 
already of type object. So no need to change them as well. 
"""
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
"""
Here "stratify=y" is used to tell scikit learn that split it using the concept 
of "Stratified Random Sampling". Fore details read about "Stratified random 
sampling" in my "Machine Learning 1st Course Guide" document. 
Actually some "status" value are yes and some are no, so there is possibility 
that in train set only those data get selected whose status value is yes (there 
is very little possibility of it but it is possible especially in case if there 
are only few "no" values ) So in such scenario model will learn to predict only
yes. Even if train set don't get all yes values but get maximum(like 90 or 95%) 
yes values then also we will get similar result. So "stratify=y" will ensure 
that both train set and test set gets yes and no value for status in the same 
proportion in which they exist in the dataset.
"""

# Build the logistic regression model :
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression();
lr.fit(X_train, y_train);

y_predict = lr.predict(X_test)


# Calculating score
score = lr.score(X_test, y_test)    

# Build the confusion matrix to compare result
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)
print(cm)
print(score)
"""
In case of replacing missing data with mean and median 
score => 0.767, cm =>
------------------------
[[ 18  40]
 [  3 124]]
------------------------

In case of simply dropping all missing value rows
score => 0.797, cm =>
------------------------
[[ 21  30]
 [  3 109]]
------------------------

So in this case dropping missing values was better in prediction but 
It doesn't mean that it's always the case. 
"""





