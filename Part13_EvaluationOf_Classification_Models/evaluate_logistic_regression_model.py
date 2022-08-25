# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 01:45:18 2022

@author: Aman Saurabh
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
    
# Build the logistic regression model :
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression();
lr.fit(X_train, y_train);

y_predict = lr.predict(X_test)


# Calculating score
score = lr.score(X_test, y_test)    

# Build the confusion matrix to compare result
from sklearn.metrics import confusion_matrix, classification_report, \
                            accuracy_score
cm = confusion_matrix(y_test, y_predict)
"""
Upto here everything is copied from logistic_regression.py file which is 
there in part8 in the project. From the following code we need to import 
"classification_report" from "sklearn.metrics". Since we have already imported 
"confusion_matrix" from the same package. So imported "classification_report" 
also above. Same for "accuracy_score". Apart from them every thing above is 
same as in logistic_regression.py file.
"""
# Create classification report for checking precision, recall etc.
cr = classification_report(y_test, y_predict) 

# Another way of check score(accuracy score) apart from "lr.score"
score_new = accuracy_score(y_test, y_predict)


