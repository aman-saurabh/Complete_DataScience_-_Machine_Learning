
"""
Data preprocessing
Created on Sun Jul 24 22:58:51 2022

@author: aman
"""

# Importing required libraries
import pandas as pd;

# Reading the CSV file
dataset = pd.read_csv("loan_small.csv");

# Accessing the datadataframe data using pandas 'iloc' function
"""
Suppose we want to get the data of first three rows for colums 
Gender and ApplicantIncome. We can do it as follows :
Note that first paramater represent "rows" and second parameter represents 
"columns"
"""
subset = dataset.iloc[0:3, 1:3];
"""
Alternate way especially useful if row or column are not in continuation
"""
subsetN = dataset.iloc[[0,1,2], [1,2]];

# Accessing the data using column names :
subsetC = dataset[['Gender', 'ApplicantIncome']]

# Accessing the data using column names and rows index :
""""
Note that here first [] (i.e [['Gender', 'ApplicantIncome']]) reprsents columns 
and second [] (i.e [0:3]) represents rows. 
And also note that here we didn't used 'iloc' function in this syntax. 
"""
subsetCR = dataset[['Gender', 'ApplicantIncome']][0:3]

# Reading the tsv file(in '.txt' format) using pandas :
"""
CSV files are "comma" delimited(i.e seperated using comma) while TSV files are
"tab" (i.e '\t') delimited. We can read tsv files also using same "read_csv" 
method of pandas using which we read csv files. But for that we have to pass 
one another argument also whose name is "sep"(representing seperator) as "\t".
We can check the list of parameters "read_csv" method(or any method in general) 
accept by selecting the function name and pressing "Control + i" 
It will also gives us same dataframe which we get while reading "csv" file.
"""
datasetN = pd.read_csv('loan_small_tsv.txt', sep='\t')


