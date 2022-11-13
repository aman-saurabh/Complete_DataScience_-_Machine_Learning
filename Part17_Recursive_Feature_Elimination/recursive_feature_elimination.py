# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 01:35:13 2022

@author: asaur
"""

# Import libraries
import pandas as pd

# Read the file
f = pd.read_csv('bank.csv')

"""
In train data we may have "duration" value which indicates call duration but in 
real data we will not have this value as we will get this value only after 
making the phone call but we will run this algorithm before making phone call
as based on it's result only we will decide whether making phone call to the 
customer will be benificial or not. So better to drop this "duration" column.    
"""
f = f.drop('duration', axis=1)

# Split the columns into dependent and independent features
X = f.iloc[:, :-1]
y = f.iloc[:, -1]

# Create dummy variables
X = pd.get_dummies(X, drop_first=True)
y = pd.get_dummies(y, drop_first=True)

# Split the dataset into train and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, \
                                                    random_state=1234)
    
# import random forest classifier
from sklearn.ensemble import RandomForestClassifier

# Default random forest object
rfc1 = RandomForestClassifier(random_state=1234)
rfc1.fit(X_train, y_train)
y_predict1 = rfc1.predict(X_test)


# Score and evaluate the model
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_predict1)
score1 = rfc1.score(X_test, y_test)

# Import Recursive feature selection
from sklearn.feature_selection import RFE
"""
It accepts many parameters. 3 main paramters are :
1.) estimator - Estimator is an algorithm such as Random forest, Decision tree,
    SVM or Multiple Linear Regression
2.) n_features_to_selectint - The number of features to select. Default is 
    None which means half of the features are selected.(It accepts absolute 
    value as integers or float between 0-1 which indicates fraction i.e 0.5 
    indicates 50% of the features to select)
3.) step - Number of features that should be eliminated in every iteration. 
    Default is 1.(It also accepts absolute value as integers or float between 
    0-1 which indicates fraction i.e 0.5 indicates 50% of the features to 
    select)
Among them estimator is required one and has no default value.
"""
# Get an estimator object
rfc2 = RandomForestClassifier(random_state=1234)
rfe = RFE(estimator=rfc2, n_features_to_select=30, step=1)

rfe.fit(X, y)

X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Fit the random forest classifier with the new train and test data
rfc2.fit(X_train_rfe, y_train)

# Test the model with new test data
y_predict = rfc2.predict(X_test_rfe)

# Score and evaluate the new model
cm_rfe = confusion_matrix(y_test, y_predict)
score_rfe = rfc2.score(X_test_rfe, y_test)
"""
You will find the score is approximately same as earlier(i.e. score1). So did 
we get any benifit? Yes, definately earlier we were getting same result with 52
features (after creating dummy variables total variable count became 52) but 
now we are getting almost same result with 30 columns only i.e. we successfully
discarded 22 unimportant features without compromizing the performance of the 
model.Isn't it great? 
"""

"""
If you open X_train_rfe or X_test_rfe you will find data of 30 most important 
features are there but feature names are not there i.e. we can't identify which 
30 features were selected. Apart from that we also don't know about the feature 
ranking and feature importance about the features of selected as well as 
non-selected features. So let's get these values.
"""
# Get column names
columns = list(X.columns)

# Get feature importance of features
feature_importance = rfc1.feature_importances_
"""
The reason we are using rfc1 and not the rfc2 here is that rfc1 has all 
columns. So we will get feature importance of all features irrespective of 
whether it was selected in recursive feature elimination or not. If we had 
used rfc2 inplace of rfc1 then we would have gotten feature importnace of only
those features which were selected in recursive feature elimination(i.e. in 
top 30 features).
"""


# Get ranking of features
ranking = rfe.ranking_
"""
You will find all features which were selected their ranking as 1 and other 
features ranking starts from 2 and increases as per their feature importance
"""

# Create the dataframe of feature names, ranking and feature importance of all
# 52 features we got after creating dummy variables.
rfe_features_details = pd.DataFrame()
"""
It will create an empty dataframe. To create a dataframe of feature names, 
ranking and feature importance we need to apply pandas "concat" function. But 
note that pandas concat function only accepts pandas dataframe or series as 
input but our 'columns' is python list, 'ranking' and 'feature_importance' 
are Array. So we will need to convert them in Pandas dataframe.  
"""
rfe_features_details = pd.concat([pd.DataFrame(columns), \
                          pd.DataFrame(ranking), \
                          pd.DataFrame(feature_importance)], axis=1)
    
rfe_features_details.columns = ["Feature Name", "Ranking", "Feature Importance"]




