# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 03:27:15 2022

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

# Create classification report for checking precision, recall etc.
cr = classification_report(y_test, y_predict) 

# Get the probabilities of the prediction. 
y_prob_all = lr.predict_proba(X_test);
# It will return probabilities for yes and no both. But for us only 
# probabilities of yes would be enough. So let's get it.
y_prob = y_prob_all[: , 1]

# Create new predictions based on given threshold
threshold = 0.8
y_predict_new = []

for i in range(0, len(y_prob)):
    if y_prob[i] > threshold:
        y_predict_new.append(1)
    else:
        y_predict_new.append(0)
        
# Again checking the score, confusion matrix and classification report for 
# this new prediction results 
score2 = accuracy_score(y_test, y_predict_new)
cm2 = confusion_matrix(y_test, y_predict_new)
cr2 = classification_report(y_test, y_predict_new)

"""
Upto here every thing is same as in "adjusting_threshold.py"(Part14).
"""

# Get the AUC and plot the ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, threshold = roc_curve(y_test, y_prob)
"""
Are you confused what we are doing and why we used y_prob? Check "roc_curve"
method details. It accepts 2 required paramters - y_true and y_score
y_true -> It is nothing but the actaul output which is y_test in our case
y_score ->it is probability estimates of the positive class, confidence values, 
or non-thresholded measure of decisions. Since y_brob is probability estimates 
of the positive class, so we used y_prob here.
And if you check what it returns - It returns 3 values.
fpr -> False positive rates [ FPR = TP / (TP + FN) ]
tpr -> True positive rates [ TPR = FP / (FP + TN) ]
thresholds -> Decreasing thresholds on the decision function used to compute 
fpr and tpr. thresholds[0] represents no instances being predicted and is 
arbitrarily set to max(y_score) + 1.
These values are used in plotting ROC curve as discussed in ROC curve section
(check ROC curve in - "Machine Learning 1st Course Guide" document for details)
"""
auc = roc_auc_score(y_test, y_prob)

# Plot ROC curve
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, linewidth=4)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Roc Curve Of Loan Prediction")
plt.grid()





