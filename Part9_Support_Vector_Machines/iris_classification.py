#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 23:19:37 2022

@author: aman
"""

# Import dataset
from sklearn import datasets

# Loading iris dataset in a variable
iris = datasets.load_iris()
"""
It(i.e. "iris" variable) will contain many values. 
To check the details of this dataset double click on "DESCR" and to view data
points double click on "data". Here note that "target" values are not part of 
"data" and is given separately. For target values double click on "target". 
To check the names of the features double click on "feature_names" and to check 
the list of target values double click on "target_names". 
"""

# Splitting dataset into test set and train set
X = iris.data
y = iris.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.3, random_state=1234, stratify=y)
    
# Implement the SVC algorithm :

# Train the SVC model
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# RBF kernel with gamma as 1
"""
Here gamma is "ɣ" which is nothing but (1/2σ^2) as we learned in Radial 
basis function(RBF) 
"""
svc = SVC(kernel='rbf', gamma=1.0)
svc.fit(X_train, y_train)

y_predict1 = svc.predict(X_test)

cm_rbf01 = confusion_matrix(y_test, y_predict1)


# RBF kernel with gamma as 10
svc = SVC(kernel='rbf', gamma=10.0)
svc.fit(X_train, y_train)

y_predict2 = svc.predict(X_test)

cm_rbf10 = confusion_matrix(y_test, y_predict2)


# Linear kernel
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

y_predict3 = svc.predict(X_test)

cm_linear = confusion_matrix(y_test, y_predict3)


# Polynomial kernel with degree as 3(default)
svc = SVC(kernel='poly')
svc.fit(X_train, y_train)

y_predict4 = svc.predict(X_test)

cm_poly_deg3 = confusion_matrix(y_test, y_predict4)

# Polynomial kernel with degree as 5
svc = SVC(kernel='poly', degree=5)
svc.fit(X_train, y_train)

y_predict5 = svc.predict(X_test)

cm_poly_deg5 = confusion_matrix(y_test, y_predict5)


# Sigmoid kernel
svc = SVC(kernel='sigmoid')
svc.fit(X_train, y_train)

y_predict6 = svc.predict(X_test)

cm_sigmoid = confusion_matrix(y_test, y_predict6)

"""
So here RBF kernel with gamma as 1 is the winner as it made only 1 wrong 
prediction. Second spot was occupied by Linear kernel as it made only 2 wrong 
predictions and third spot was occupied by Polynomial kernel(tie between 
Polynomial kernel with degree 3 and Polynomial kernel with degree 3) and fourth 
spot was occupied by RBF kernel with gamma as 10. And sigmoid kernel was the 
worse performing kernel function in this case as it has only 3 correct 
prediction out of 45. 
But it doesn't mean that sigmoid kernel is the worse kernel function. Actually 
it was not suitable for this dataset. For different datasets different kernel 
and different values of gamma(only in case of RBF kernel) can be best suited. 
So better try with different kernel and different values of gamma. Select the 
kernel and gamma value for which it makes best prediction.
"""





