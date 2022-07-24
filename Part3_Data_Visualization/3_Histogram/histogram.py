#Importing library for data visualization or plotting
import matplotlib.pyplot as plt

f= open('agedata.csv')
# Before running it make sure that your current working directory is updated.
agefile = f.readlines()

#Integer list
age_list = []

for records in agefile:
    age_list.append(int(records))
    
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Plot the histogram 
plt.title('Age Histogram')
plt.xlabel('Group')
plt.ylabel('Age')

plt.hist(age_list, bins, histtype='bar', rwidth=0.9)