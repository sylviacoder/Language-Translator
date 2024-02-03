#the findall function returns a list that has found a match in a string
import re

#returns a list containing every occurence of "ai"

txt = "The rain in Spain"
x = re.findall("ai", txt)
print(x) 

#The list contains the matches in the order they are found. If no matches are found, an empty list is returned
import re

txt = "The rain in Spain"

#Check if "Portugal" is in the string:

x = re.findall("Portugal", txt)
print(x)

if (x):
  print("Yes, there is at least one match!")
else:
  print("No match")

#The search() function searches the string for a match, and returns a Match object if there is a match. 
#If there is more than one match, only the first occurrence of the match will be returned
#to search for white space characters in a string
import re

txt = "The rain in Spain"
x = re.search("\s", txt)

print("The first white-space character is located in position:", x.start())

#If no matches are found, the value None is returned
import re

txt = "The rain in Spain"
x = re.search("Portugal", txt)
print(x)

#The split() function returns a list where the string has been split at each match
import re

#Split the string at every white-space character:

txt = "The rain in Spain"
x = re.split("\s", txt)
print(x)

#You can control the number of occurrences by specifying the maxsplit parameter
import re

txt = "The rain in Spain"
x = re.split("\s", txt, 1)
print(x)

#The sub() function replaces the matches with the text of your choice
import re

#Replace all white-space characters with the digit "9":

txt = "The rain in Spain"
x = re.sub("\s", "9", txt)
print(x)

#You can control the number of replacements by specifying the count parameter
import re

txt = "The rain in Spain"
x = re.sub("\s", "9", txt, 2)
print(x)



