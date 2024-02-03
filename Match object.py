#Match object
#a match object is an object containing information about the search and the result
#if there is no match, the value None will be returned  instead of the match object
import re

#The search() function returns a Match object:

txt = "The rain in Spain"
x = re.search("ai", txt)
print(x)
