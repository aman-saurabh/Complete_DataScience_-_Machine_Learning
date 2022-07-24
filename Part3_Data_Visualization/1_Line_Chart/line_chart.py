#Plotting Line Chart
"""
Spyder setting to display the plot in a seperate pop-out window instead of variable explorer.
Tools -> Preferences -> IPython console -> Then go to "Graphichs" tab 
Then select Qt5 or QT4 in "Backend dropdown".
If you select "inline" then also plot will be shown to you but in the variable explorer. 
"""
#Importing library for data visualization or plotting
import matplotlib.pyplot as plt
# Week Days
x_days = [1, 2, 3, 4, 5]
# Prices of stock1
y_price1 = [9, 9.5, 10.1, 10, 12]
# Prices of stock2
y_price2 = [11, 12, 10.5, 11.5, 12.5]

# Define the chart elements
plt.title('Stock Movement')
plt.xlabel('Week Days')
plt.ylabel('Price in USD')

# Plotting the actual graph
# Plotting first stock
plt.plot(x_days, y_price1, label='Stock 1')
# Plotting second stock
plt.plot(x_days, y_price2, label='Stock 2')

#Defining the legend position and font
plt.legend(loc=2, fontsize=12)
# loc stands for location
# 1 means top-left corner, 2 means top-right corner, 
# 3 means bottom-left corner, 2 means bottom-right corner

#Displaying the graph
plt.show()



 