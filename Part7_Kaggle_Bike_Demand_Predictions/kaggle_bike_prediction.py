#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 15:48:56 2022

@author: aman
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#Reading hour.csv file
bikes = pd.read_csv("hour.csv");

# ------------------------------------------------------------------
# Prelim Analysis and Feature selection
# ------------------------------------------------------------------
# Droping irrelevant data
bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index', 'date', 'casual', 'registered'], axis=1)

# Basic check for missing values
bikes_prep.isnull().sum()

# ------------------------------------------------------------------
# Visualizing data and drawing conclusions based on that
# ------------------------------------------------------------------
# Visualize the data using pandas histogram
bikes_prep.hist(rwidth=0.9)
"""
If you run above code you will see data of one disgram is overlapping with 
other. To get rid of that problem let's call tight_layout() method of pyplot.
rwidth - The relative width of the bars as a fraction of the bin width.
bars - It is nothing but thick blue verticle lines whichwe see in histogram.
We can set numer of bars also by passing "bars=any_number" as argument in 
hist() function. Default number of bars for hist() function is 10 i.e if 
we don't pass "bars=any_number" then 10 bars will be created in the histogram.
"""

plt.tight_layout()


# Data visualization of continuous features with respect to demand. 
# Continuous features(i.e numerical columns) are-temp,atemp,humidity & windspeed
"""
Note that "atemp" is nothing but 'feel like temperatue'. Sometimes we feel like
the temperature is 35 degree C but when we check we get it is 30 degree C. The 
reason behind it can be humidity, atmosphere etc. Such temperature is known as 
'feel like temperature'. 
"""
plt.subplot(2, 2, 1)
plt.title("Temp vs Demand")
plt.scatter(bikes_prep['temp'], bikes_prep['demand'], s=2, c='g')

plt.subplot(2, 2, 2)
plt.title("Atemp vs Demand")
plt.scatter(bikes_prep['atemp'], bikes_prep['demand'], s=2, c='b')

plt.subplot(2, 2, 3)
plt.title("Humidity vs Demand")
plt.scatter(bikes_prep['humidity'], bikes_prep['demand'], s=2, c='m')

plt.subplot(2, 2, 4)
plt.title("Windspeed vs Demand")
plt.scatter(bikes_prep['windspeed'], bikes_prep['demand'], s=2, c='c')

plt.tight_layout()

# Data visualization of categorical features with respect to demand. 
# For season demand :- 
"""
All season have multiple entry in the dataset so we need to find the average 
demand for every season and the list of all unique seasons to plot the graph of 
season vs demand.
"""
"""
# Create list of unique seasons
cat_list = bikes_prep['season'].unique()  

# Create average demand per season
cat_average = bikes_prep.groupby('season').mean()['demand']

# Plotting the graph of season vs average demand 
colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color=colors)
# It will plot graph with 'color' attribute also but in that case all data 
# will be shown in same color which is boring to see. So we added color to it.

#Now the problem is this way we can show only one graph at a time. To 
#show multiple graphs at the same time we need to show it as subplot. 
#So let's add it and code for other features as part of subplots.
"""
colors = ['g', 'r', 'm', 'b']

plt.subplot(3,3,1)
plt.title('Average Demand per Season')
cat_list = bikes_prep['season'].unique()
cat_average = bikes_prep.groupby('season').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,2)
plt.title('Average Demand per month')
cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,3)
plt.title('Average Demand per Holiday')
cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,4)
plt.title('Average Demand per Weekday')
cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,5)
plt.title('Average Demand per Year')
cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,6)
plt.title('Average Demand per hour')
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,7)
plt.title('Average Demand per Workingday')
cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.subplot(3,3,8)
plt.title('Average Demand per Weather')
cat_list = bikes_prep['weather'].unique()
cat_average = bikes_prep.groupby('weather').mean()['demand']
plt.bar(cat_list, cat_average, color=colors)

plt.tight_layout()

"""
Important conclusions we can draw from visualizing plots of continuous and 
categorical data and histogram.
Histogram :-
Dependent variable "demand" is not normally distributed (i.e. it's plot is not 
bell shaped)
So wew will require to do some transformation in it so that we get a normally 
distributed demand data like log transformation, min-max transformation etc.
We will see that at appropriate step. 


Continuous data plots :-
1.) temp and atemp vs demand :-
temp and demand have some direct correlation. And at the same time atemp and 
demand also have almost similar corelation.So there is chance of correlation 
between 'temp' and 'atemp'. So we need to check condition for Multicolinearlity
between them.
2.) windspeed and humidity vs demand :-
We are not clear about the correlation between these and demand from the graph. 
So we can't draw any conclusive decision at this stage and we need some more 
statical analysis.


Categorical data plots :-
1.) Features we can drop :- As these featured don't show significant changes in 
demand across its values.
- year             (- Data is insufficient(only 2 data))
- weekday          (- Demand is almost same for all weekdays)
- workingday       (- Demand is almost same for workingdays as well as for 
                    non-workingdays)
2.) Average demand per hour :-
We can see the bike demand is higher at 8am and 6-7pm.It's obvious also since 
around 8am people generally move for office and around 6-7pm people generally 
leave from office. So need to make sure the availablity of maximum number of 
bicycles nearby public transport around 8pm and nearby office premises around 
6-7pm.  
"""
# ------------------------------------------------------------------
# Check for outliers
# ------------------------------------------------------------------
bikes_prep['demand'].describe()
"""
describe() method details :-
The describe() method returns description of the data in the DataFrame.

If the DataFrame contains numerical data, the description contains these 
information for each column:

count - The number of not-empty values.
mean - The average (mean) value.
std - The standard deviation.
min - the minimum value.
25% - The 25% percentile*.
50% - The 50% percentile*.
75% - The 75% percentile*.
max - the maximum value.

*Percentile meaning: how many of the values are less than the given percentile.

Since we have applied "describe()" method for "demand" column only so in our 
case above mention data is related to "demand" column only. i.e it will return 
'mean', 'standard deviation' etc. for "demand" column only.

In the output look at the following data :-
---------------------------------
25%         40.000000
50%        142.000000
75%        281.000000
---------------------------------
It shows the mean of Interquartile range(IQR) - Interquartile range(IQR) 
describes the middle 50% of values when ordered from lowest to highest.
Check "Interquartile range(IQR)" in "Data science 2022" document for details.

So from here we can see that the value of "demand" for 50% of the data is 
between 40 and 281 which is fairly away from the max(977) and min(1) values.
So there is a high chance of outliers. 
"""
bikes_prep['demand'].quantile([0.05, 0.1, 0.15, 0.9, 0.95, 0.99])
"""
Quantile :-.
The word “quantile” comes from the word quantity. In simple terms, a quantile 
is where a sample is divided into equal-sized subgroups. It can also refer to 
dividing a probability distribution into areas of equal probability.
A quantile determines how many values in a distribution are above or below a 
certain limit. Special quantiles are the quartile(represents quarter), the 
quintile(represents fifth part) and percentiles(represents hundredth part).
An example :
If we divide a distribution into four equal portions, we will speak of 
"four quartiles". if we arrange all values in ascending order then the first 
25% values will represent first quartile. In a graphical representation, 
it corresponds to 25% of the total area of a distribution. The two lower 
quartiles comprise 50% of all distribution values. The interquartile range 
between the first and third quartile equals the range in which 50% of all 
values lie that are distributed around the mean. 

Pandas quantile :-
Pandas quantile() method is used to determine the value below which given 
percentile of data will fall. Default is 50 percentile i.e if you don't pass 
any value for percentile i.e if you call it as follows :
---------------------------------------------
bikes_prep['demand'].quantile()
---------------------------------------------
then it will return the value below which 50% of the data will fall.
However you can pass other values of quantile also. For example - quantile for
5 percentile.
---------------------------------------------
bikes_prep['demand'].quantile(0.05)
---------------------------------------------
It returns '5' it means the value of "demand" for the 5% of the data is below 
5.  

Here from the following command(which is used in actual program also) :-
----------------------------------------------------------------------
bikes_prep['demand'].quantile([0.05, 0.1, 0.15, 0.9, 0.95, 0.99])
----------------------------------------------------------------------
We are calculating "quantiles" for 5%, 10%, 15%, 90%, 95% and 99% in one go.
It's output is 
--------------------------------------
0.05      5.00
0.10      9.00
0.15     16.00
0.90    451.20
0.95    563.10
0.99    782.22
--------------------------------------
So from here we can conclude that 5% of the time demand is below 5 and 1% of 
the time demand is above 782. These values are considered as outliers and 
should be removed from the dataset.

We will see how to remove these outliers later. Before that let's check for 
other assumptions for "Multiple Linear Regression". 
"""

#Linearity using correlation coefficient matrix using pandas corr() method.
correlation = bikes_prep[['temp','atemp','humidity',
                          'windspeed','demand']].corr()
"""
If we print the correlation variable we get result as follows :
print(correlation)
--------------------------------------------------------------------
               temp     atemp  humidity  windspeed    demand
temp       1.000000  0.987672 -0.069881  -0.023125  0.404772
atemp      0.987672  1.000000 -0.051918  -0.062336  0.400929
humidity  -0.069881 -0.051918  1.000000  -0.290105 -0.322911
windspeed -0.023125 -0.062336 -0.290105   1.000000  0.093234
demand     0.404772  0.400929 -0.322911   0.093234  1.000000
--------------------------------------------------------------------
We can see that all values in diagonal line is 1(i.e. highest) which shows very 
high correlation but note that diagonal line is representing correlation with 
itself so ignore these value.
Now look at correlation coefficient of temp with atemp. It is 0.987 which is 
again very high. From visulaizing graph of temp vs demand and atemp vs demand 
also we had suspected that. So from here it is confirmed and hence we need to 
drop atemp column. Also see that temp correlation of temp with demand. It is 
0.4 i.e 40% which is quiet high and it confirms temp data is relavent for 
predicting demand so we are going to keep it.
Now let's look at the correlation coefficient of humidity with other columns.
Correlation coefficient is not very high for any other independent columns and 
it is -0.32 i.e 32% with demand. So again it seems relavent for prediction of 
demand and we are going to keep it also.
Again look at the correlation coefficient of windspeed with other columns.
Correlation coefficient is not very high for any other independent columns but 
the correlation coefficient with demand is 0.09 i.e 9% which is very less and 
seems irrelevant for the prediction of demand and hence we are going to drop
this column.
So from here we have analyzed that we are going to drop "atemp" and "windspeed"
And remember after visualizing graph of categorical data w.r.t demand we had 
decided that we will drop year, workingday, weekday columns. So let's drop all 
these columns. 
"""
bikes_prep = bikes_prep.drop(['weekday', 'workingday', 'year', 
                              'atemp', 'windspeed'], axis=1)


# Check for the autocorrelation in demand using pandas "acorr()" method
"""
Since acorr() function accepts only float(i.e. decimal) values so let's 
convert it into float format.
"""
df1 = pd.to_numeric(bikes_prep['demand'], downcast='float')
# Plotting autocorrelation graph
plt.acorr(df1, maxlags=12)
"""
Autocorrelation occurs in timeseries data and demand is registered against 
hourly(represented by 'hour' column) data also. So chances of autocorrelation 
is in hourly data only. Hence we picked maxlags as 12 representing 12 hours in 
24 hours day. 

From the graph we can see that autocorrelation coefficient is very high for 
last 5 values(greater than usual threshold value of 0.05 i.e. 5%). However we 
are considering only last 3 values as they are significantly high(greater than 
0.8 i.e. 8%).
Here we are talking only about previous(last) values as always next value is \
dependent on previous value in autocorrelation and timeseries.
Since there is high autocorrelation in the demand feature so we should remove 
it but since "demand" is the dependent feature so we can't remove this feature, 
we will handle to handle it. 
We will handle it but in visualizing histogram we have seen that 'demand' 
doesn't have normal distribution also so we need to handle normality also so 
let's first handle the normality issue of demand and then we will handle its
autocorrelation issue.  
"""

# Handling the issue of normality in demand.
# let's draw the histogram of 'demand' again to analyze it
bikes_prep['demand'].hist(rwidth=0.9)
"""
By looking the histogram of demand we can say that it doesn't show normal 
distribution but now the question is which kind of distribution it shows:
It resembles very close to the "Log-Normal distribution". Log-Normal 
distribution is a kind of distribution where the log values of the actual 
value of the record shows normal distribution. So if we convert the demand 
values with its corresponding 'log' values then we will get the normal 
distribution for demand feature.     
"""

# Storing demand column in a variable "dt1"
dt1 = bikes_prep['demand']
# Storing demand column in a variable "dt2"
dt2 = np.log(dt1)
"""
Let's plot the histogram for "dt1" and "dt2" again to see if we get normal 
distribution in "dt2". dt1 histogram will be exactly same as in previous 
case. Here we are plotting it again together with "dt2" to compare both 
graphs together. 
"""
plt.figure()
dt1.hist(rwidth=0.9, bins=20)

plt.figure()
dt2.hist(rwidth=0.9, bins=20)
"""
Select both together and execute together.
Here plt.figure() is used to create different figures for both graphs (i.e. 
different graphs with different figure numbers like - 'Figure1', 'Figure2'). 
If you don't apply plt.figure() then only one graph will be shown.
Here "bins" means number of bins (represented by thick blue vertical line in the 
histogram). Default is 10 i.e if you don't specify bins then in the histogram 
there will be only 10 bins(thick blue vertical line).
From the graph of dt2(Figure2) you can see that now it has some resemblance 
with Normal distribution. However it is still not perfectly normally 
distributed(specially on lower side). It's because of the presence of outliers. 

Now let's replace the "demand" column of bikes_prep with dt2 as we need to 
replace the demand values with their corresponding log values.And dt2 has 
corresponding log values. Optionally you can apply np.log() function directly 
on "bikes_prep['demand']" also. Let's do it :
"""
bikes_prep['demand'] = np.log(bikes_prep['demand'])


# handling the issue of autocorrelation in demand :-
"""
Here we are using pandas "shift()" and "to_frame()" methods. Check following 
file of this project for details :
*************************************************************************
Part6_Multiple_Linear_Regression/2_autocorrelation/autocorrelation.py 
*************************************************************************
Consider the case of the following :
---------------------------------------------------
t_1 = bikes_prep['demand'].shift(+1).to_frame()
---------------------------------------------------
It create a lag of 1 and will return a dataframe with one column 'demand'. 
But we want 3 such dataframe representing 1, 2 and 3 lags respectively. Since 
we had decided to handle the autocorrelation of last 3 data. In such case all 
dataframes will have one column named 'demand'. And we will also need to 
include these dataframes in original dataframe(i.e bikes_prep) as column which 
already has a 'demand' column. So we will need to change the name of columns in
these dataframes. We can do that as follows :   
---------------------------------------------------
t_1.columns = ['t-1']
---------------------------------------------------
Here we are giving the column name as 't-1'. So in the dataframe t_1 now the 
name of the 'demand' column will be replaced by 't-1'. 
"""
# For lag of 1
t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']

# For lag of 2
t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']

# For lag of 3
t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

# Adding these dataframes as columns in bikes_prep. But here instead of 
# changing bikes_prep we are creating a new dataframe.
bikes_prep_lag = pd.concat([bikes_prep, t_1, t_2, t_3], axis=1)
"""
If you see "bikes_prep_lag" you will find lag values at the first 3 columns as 
lagged data creates nan values equal to the number of lag. Since here we have 
created till lag of 3. So t-3 column will have 3 nan values, similarly t-2 will 
have 2 nan values and t-1 will have 1. So let's drop rows with nan values.
"""
bikes_prep_lag = bikes_prep_lag.dropna()

# Create dummy variables :
# dummy_df = pd.get_dummies(bikes_prep_lag, drop_first=True)
"""
pandas "get_dummies()" method is used for one hot encoding i.e to create dummy 
variable and here we are passing drop_first=True to drop first dummy column 
from every set of dummy variables to avoid dummy variable trap and correlation 
issues.
But if you run the above line you will find dummy variables are not created.
There is something wrong in above code and hence we commented out that. 
It's pandas get_dummies create dummies only for columns whose datatype is 
'category'. But if you see the datatypes of all columns you will find no one 
has datatype of category. So we need to convert datatypes of categorical 
columns to 'category'
"""
# Check columns datatypes.
bikes_prep_lag.dtypes
"""
From here we can see that categorical column are 'season', 'month', 'hour', 
'holiday' and 'weather' and datatype of all these columns are int64. So let's 
convert their datatype to 'category'.
"""
bikes_prep_lag['season'] = bikes_prep_lag['season'].astype('category')
bikes_prep_lag['month'] = bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour'] = bikes_prep_lag['hour'].astype('category')
bikes_prep_lag['holiday'] = bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather'] = bikes_prep_lag['weather'].astype('category')

# Now let's create dummy variables :
dummy_df = pd.get_dummies(bikes_prep_lag, drop_first=True)
"""
Now you will see the number of columns is increased from 8 (in bikes_prep_lag)  
to 47 in dummy_df
"""

# Splitting the dataset into train dataset and test dataset - Time series data
"""
We have already learnt to split data into train dataset and test dataset using 
scikit learn(i.e. sklearn.model_selection) moduletrain_test_split() method as 
follows :
-------------------------------------------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, random_state=123)
-------------------------------------------------------------------
It will randomly pick 20% data and put that in test set and rest 80% in train 
set. But here the dataset is a time series data and there is an autocorrelation
between values of "demand". So if we pick random data from the middle then 
these autocorrelation will be lost and we may end up with wrong prediction. 
So we need to select consecutive data either from the starting or from the end 
as test data and rest will be train data. We can pick consecutive data from the 
middle also as test data but it is not recommended as in this case also 
autocorrelation may be disturbed to some extent.
"""

# Creating X(independent features) and Y(dependent feature)
y = bikes_prep_lag[['demand']]
# Double square bracket to get data in the form of dataframe
X = bikes_prep_lag.drop(['demand'], axis=1)

# Create size for 70% of data
tr_size = 0.7 * len(X)
# It will return value in float, so let's convert it into int
tr_size = int(tr_size)

# Create X_train, X_test, Y_train and Y_test for time-series data
X_train = X.values[0 : tr_size]
X_test = X.values[tr_size : len(X)]

y_train = y.values[0 : tr_size]
y_test = y.values[tr_size : len(y)]

# Train the multiple linear regression model
from sklearn.linear_model import LinearRegression

# Creating regressor of LinearRegression class
std_reg = LinearRegression()

# Train and fit the training data
std_reg.fit(X_train, y_train)

# R-squared and the equation of the line
mlr_score_train = std_reg.score(X_train, y_train)
mlr_score_test = std_reg.score(X_test, y_test)
"""
We get value of mlr_score_train = 0.888 and mlr_score_test = 0.862 which is 
quiet high(very close to 1). So it means our model has done a good job.
"""

# Creating prediction - let's predict the values of Y from the test data
y_predict = std_reg.predict(X_test) 


# Calculate Root mean squared error(RMSE) :-
# To know how much error our model has made
from sklearn.metrics import mean_squared_error
import math
mlr_rmse = math.sqrt(mean_squared_error(y_test, y_predict))







