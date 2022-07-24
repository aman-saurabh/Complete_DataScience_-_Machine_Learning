#Importing library for data visualization or plotting
import matplotlib.pyplot as plt

f= open('salesdata.csv')
# Before running it make sure that your current working directory is updated.
salesfile = f.readlines()

#Integer list
sales_list = []

for records in salesfile:
    sales_list.append(int(records))

plt.title('Box plot of sales')
plt.boxplot(sales_list)
plt.show()