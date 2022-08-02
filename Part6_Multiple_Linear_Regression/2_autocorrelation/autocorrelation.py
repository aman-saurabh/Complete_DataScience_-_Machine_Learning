#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autocorrelation
Created on Mon Aug  1 01:52:20 2022

@author: aman
"""
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("03corr.csv")

# Plotting autocorrelation
# plt.acorr(dataset['t0'], maxlags=10)
"""
Above code will throw error like below :-
TypeError: ufunc 'true_divide' output (typecode 'd') could not be coerced to 
provided output parameter (typecode 'l') according to the casting rule 
''same_kind''

It is because pyplot's "acorr" function expects values in decimal(i.e. float) 
format but in the dataset it is in integer format. So we need to convert it 
into decimal format. 
"""

dt = dataset.copy()

dt['t0'] = pd.to_numeric(dt['t0'], downcast="float")

# Plotting autocorrelation
plt.acorr(dt['t0'], maxlags=10)


#================== Additional Optional Information =======================
# Creating shifted or timelag data :
# Creating 2 datasets with timelag of 1 and 2 respectively :
t_1 = dt['t0'].shift(+1).to_frame()
"""
Here ".shift()" and ".to_frame()" are pandas functions.
".shift(+1)" will shift the data from +1
and ".to_frame()" will simply convert the shifted data into a dataframe.
Similarly we can find t_2 for timelag of 2 also.
"""
t_2 = dt['t0'].shift(+2).to_frame()









