#regular expression
#RegEx can be used to check if a string contains the specified search pattern
#the built in re is used to check regular exressions, it can be used in instances of searching for specific things on the web and even in game applications
#to do this, we import the re module as seen in line 5
import re
#it is only when you have imported the re module can you now start using regular expressions

txt = "The rain in Spain"
x = re.search("^The.*Spain$", txt)

if x:
    print("YES! We have a match!")
else:
    print("No match")

#in real life scenerios, regex can be applied in apps like spotify where it finds artists and can bring out the list of their songs
#the re module offers other functions that allows us to search a string to match 