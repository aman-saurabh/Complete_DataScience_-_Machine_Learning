# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 02:20:57 2022

@author: Aman Saurabh
"""
"""
In the previous model we found that the precision for the model was 
0.78 and recall was 0.97(which is a very good recall). But we have learnt that
in some cases precision have greater importance(where Flase positives have 
greater importance - i.e. we don't want our model to predict an output as 
positive if it is actually negative, however if it predicts an output as 
negative if it is actually positive that is tolerable). Similarly in some
cases recall have greater importance(where Flase Negatives have 
greater importance - i.e. we don't want our model to predict an output as 
negative if it is actually positive, however if it predicts an output as 
positive if it is actually negative that is tolerable). 
Suppose in this case precision is important and we want precision to be around 
0.95(or 95%). How can we achieve this?
Recall how Logistic regression predicts output. It calculates the 
probabilities of yes(denoted by P while probability of no is denoted by 1 - P) 
and compare the probability with the threshold value(generally 0.5) And 
returns the value as true if P > threshold and false if P < threshold.
Sklearn LogisticRegression model also considers the threshold value as 0.5. 
We can get higher precision(or even higher recall also) by tweaking the 
threshold value but unfortunately Sklearn LogisticRegression doesn't give us 
option to set the threshold value. But luckily it gives us option to get the 
probabilities of true and false using LogisticRegression    
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

# Here we can see that the value of False positives have gone down from 30 to 
# 14 and hence the precision has gone up from 0.78 to 0.84. which is still less
# than our target of 0.95, so change the threshold and run the model again 
# and do it until we get the most nearest value from our target if not above 
# target value. 
# But note that doing so will impact on recall also and updating threshold 
# from 0.5 to 0.8 reduced the recall from 0.97 to 0.68. So if you further 
# increase the threshold it will further decrease the recall. So take care that
# recall value don't reach a very less value as it is also an important 
# parameter. 




   
