# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 19:57:00 2022

@author: asaur
"""
# Import libraries
import pandas as pd

# Read the file
f = pd.read_csv('Students2.csv')

# Split the columns into Depndent(Y) and Independent(X) features
X = f.iloc[:, :-1]
y = f.iloc[:, -1]

# Import various select transforms along with the f_regression mode
from sklearn.feature_selection import SelectKBest, \
                                      SelectPercentile, \
                                      GenericUnivariateSelect, \
                                      f_regression
    
# Implement SelectKBest
selectorK = SelectKBest(score_func=f_regression, k=3)

# Storing k(as mentioned in selectorK) best features in X_k variable
X_k = selectorK.fit_transform(X, y)
"""
If you look at X_k you will find there are three columns of data. Actually 
these are the selected 3 best columns as we had set k = 3 in the selectorK. But
you will find it didn't printed the colmns names which are selected as the 
table header. So we can't identify which 3 features are selected unless we 
match the column data with the original X. Apart from that we also don't know 
what was the f-score and p-values of these columns. So let's check all these 
information.   
"""

# Get the f_score and p_values
f_score = selectorK.scores_
p_values = selectorK.pvalues_

# Print the table of features, f-score and p-values
columns = list(X.columns)
print(" ")
print(" ")
print(" ")

print("    Features    ", "    F-scores    ", "    P-values    ")
print("  ------------  ", "  ------------  ", "  ------------  ")

for i in range(0, len(columns)):
    f1 = "%4.2f" % f_score[i]
    p1 = "%2.6f" % p_values[i]
    print("   ", columns[i].ljust(12), " ", f1.rjust(8),"         ", p1.rjust(8))
    
"""
From the table we can see that the best 3 features are "hours", "sHours" and 
"calories". But here there were total 6 columns only so we could find it 
manually so easily but what if there were thousands of columns. So it would be 
better if we can directly find the column names of best k(in this case 3) 
columns which were selected in X_k  
"""
cols = selectorK.get_support(indices=True)
print(cols)
"""
cols will return the selected column indeces(plural of index) as [0,1,5].
Since first(index=0), second(index=1) and sixth(index=5) columns are selected.

If we set "indices=False" then it will return result as follows :
[ True  True False False False  True]
Here "True" indicates column is selected and "False" indicate column is not 
selected. So first(index=0), second(index=1) and sixth(index=5) are selected 
and rest are not selected.

Here again we got selected columns indices but not the selected columns names.
But since we have selected columns indices, so we can get those columns names 
using the following code.
"""
selected_cols = X.columns[cols].tolist()

print(selected_cols)

# Implement SelectPercentile
selectorP = SelectPercentile(score_func=f_regression, percentile=50)

X_p = selectorP.fit_transform(X, y)
"""
Again it will return the best 3 columns as here total column count is 6 and we 
hd set percentile=50%. So 50% of 6 columns is 3 columns only.
Again it will not print the selected columns names, p_values and f_socres, but
we can get them as we got in previous section i.e. in SelectKBest.
"""

# Implement GenericUnivariateSelect 
# For SelectKBest
selectorG1 = GenericUnivariateSelect(score_func=f_regression,
                                     mode='k_best',
                                     param=3)

X_g1 = selectorG1.fit_transform(X, y)

# For SelectPercentile
selectorG2 = GenericUnivariateSelect(score_func=f_regression,
                                     mode='percentile',
                                     param=50)

X_g2 = selectorG2.fit_transform(X, y)






