#Importing library for data visualization or plotting
import matplotlib.pyplot as plt

x_cities = ['New York', 'London', 'Dubai', 'New Delhi', 'Tokyo']
y_temp = [75, 65, 105, 98, 90]

# Define the chart elements
plt.title('Temperature variations')
plt.xlabel('Cities')
plt.ylabel('Temperature')

# Plotting the actual graph
plt.bar(x_cities, y_temp)
# Notice here that we have not used plt.plot(), instead we used 
# plt.bar() for bar chart. 

# Displaying the plot
plt.show()