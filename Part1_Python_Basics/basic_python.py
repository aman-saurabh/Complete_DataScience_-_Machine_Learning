# My first python code
print("Hello World!");

a = "Hello Python!"
print(a)

for i in range(0, 10, 2):
    print(i)
else:
    print("Outside of loop")
    
list1 = ["Jitesh", "John", "Aliya"]
print(list1[2])

#Operations on list 
# Add element 
list1.append("Lisa")
print(list1, "after adding")

# Update element
list1[1] = "Rocky"
print(list1, "after updating")

# Delete an element
del list1[1]
print(list1, "after deleting")

# Get number of elements in the list
len(list1)
# When you run this line of code it will automatically print the value in 
# console. So you don't need to write print statement explicitely.

# Add(concatinate) two lists
list2 = ["Mango", "Jitesh", "Lion", "Potato"]
con_list = list1 + list2
print(con_list)
# Since 'Jitesh' is there in list1 and list2 both, so in con_list 
# there will be 2 'Jitesh'.  

# Sort list
# Ascending order(default)
con_list.sort()
print(con_list, "ascending order")

# Descending order
con_list.sort(reverse=True)
print(con_list, "descending order")

# Multidimensional list(List within a list)
two_d_list = [["Jitesh", "Lee", "Lisa"], [121, 346, 273]]
print(two_d_list[0])
print(two_d_list[1][0])

# Slicing multidimentional list
# Long method :- Using traditional programming approach
sublist1 = []
for l1 in two_d_list:
    idx1 = two_d_list.index(l1)
    sublist1.append([])
    for l2 in l1:
        idx2 = l1.index(l2)
        if (idx2 > 0):
            sublist1[idx1].append(l2)
print(sublist1)


# Short method :- Using python list slicing concept
sublist2 = []
for l1 in two_d_list:
    sublist2.append(l1[1:3])
print(sublist2)
    
# Tuple in python :-
"""
Tuples are similar to lists in python and like lists, tupleas are also
used to store multiple values within it (i.e in a single variable). 
But there are some differences in Tuple and List :
1.) Tuple is represented by small brackets(i.e parenthesis) while lists 
    with square brackets.
2.) Tuples are immutable while lists are mutable.
3.) Tuples are faster to process in comparison to lists. So if you don't 
    need to update any list after it's creation then use tuple instead.
"""
tup1 = (1,2,3) 
print(tup1[1])

# Adding, removing and updating values in tuple is not allowed in tuple 
# as tuples are immutable. So following lines will throw error. 
# tup1[1] = 243
# tup1.append(63)
# del tup1[0]

# Get number of elements in the tuple
len(tup1)

# Addition(Concatination) of tuples: 
con_tup = tup1 + ("Aman", "Mango", "Dear")
"""
We have learnt that tuples are immutable but if we try to store combined 
value in tup1 itself instead of con_tup (i.e in a new variable) then
also it works and we don't get any error. i.e following line will
not throw error. It's because in such scenario we are not updating 
the existing tuple represented by tup1 but instead we are reassigning 
tup1 variable with a new tuple. So it will not throw error. 
"""
tup1 = tup1 + ("Aman", "Mango", "Dear")

# Dictionary in python
"""
Dictionary is collection of key-value pairs i.e it is used to store data in 
the form of key-value pairs. It is similar to Map of other languages.
In dictionary values can be accessed using the key.
Dictionaries are represented by curly braces(i.e {})
"""
address = { 'Street' : 'Ang Street',
           'City': 'Munger',
           'State': 'Bihar',
           'Country': 'India'}

print(address['Street'])

# Checking if the dictionary contains a key
pin_key = 'Pin'
print(pin_key in address)
# You can directly use string value here also like -
# print('Pin' in address)

# Get all keys of a dictionary
address.keys()
# Running loop over all keys 
keys = address.keys()
for k in keys:
    print(address[k])
else:
    print("All values printed");
    
# Get all values of a dictionary
address.values()

# Updating value for key
address['Street'] = 'Mithila Street'

# Adding a new key value pair
address['Pin'] = 811201

# Deleting a key-value pair
del address['Pin']

# Get the length of a dictionary i.e number of key value pairs
len(address)

# Convert the dictionary into string
str(address)
# i.e here str() method is working in the similar way how JSON.stringify()
# works for objects in JavaScript.

# Sets in Python :-
"""
Sets in python is similar to list, and is used to store multiple values within 
it(i.e in a single variable). But like tuples, sets also has some differences
with Lists :
1.) Sets are represented by curly braces while lists are represented by
    Square brackets.
2.) Sets cannot have multiple occurrences of the same element while 
    lists can have.
3.) Lists are ordered but Sets are unordered i.e in list elements exist in 
    the same order in which they were inserted but that is not true for sets.
    i.e in sets elements can be stored in any random order and hence concept 
    of index is not valid for sets.
Few more importants points :-
1.) Like lists sets are also mutable.
"""
set1 = {"Aman", "Saniya", "Michael"}
print(set1)

# Adding element to set
set1.add("Jackie")
print(set1)

# Deleting element from set
set1.remove("Aman")
print(set1)

# Deleting last element of the set
set1.pop()
print(set1)

# 
list1.pop()

# -------------------------------------------------
# Python file handling
# -------------------------------------------------
cityTempRead = open('citytemp.csv', 'r')
# 'r' mode represents 'Open for reading'. It is the default mode i.e.
# if you don't specify any mode then also file will be open in 'r' mode.
# You can check the details of all modes by checking the open method details
# by selecting the 'open' keyword and clicking on the 'ctrl + i' button 
# thereafter

cityTempWrite = open('citytemp.csv', 'a')
# 'a' mode represents 'open for writing, appending to the end of the file if it exists'





















