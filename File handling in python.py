#File handling in python
#It can be used to create things like audiobook or pdf to text
#to do this, we use the open function. The parameters that follows it is the file name and mode
#There are four different methods to operate file handling
#line 6 shows the format
#open(filename, 'mode')
#there are different modes the file takes
#"r" = read
#"a" = append
#"w" = write
#"x" = create
#the read gives an error if the file does not exist
#the append mode does not give an error instead it helps you to create the file if it does not exist
#the write mode helps us to write something in the file. It also has an override function. if the file does not exist, it helps create one replacing the old one
#For the create, if the file exists it gives an error 


#Append
#To write to an existing file, you use the append method which uses the "a"
#my_file = open('Polymorphism.py', "a")
#my_file.write('This is my file handling class')
#my_file.close()


#READ
#if you want to read just some parts of your file include the specification in your bracket
#if you want to print the first line then you use readline()
#my_file = open('Polymorphism.py', "r")
#print(my_file.read())
#my_file.close()

#...when it does not exist
#file2 = open('file_assist.txt', "a")
#file2.write('This is a new file')
#file2.close()

#file2 = open('file_assist.txt', "r")
#print(file2.read())
#file2.close()

#WRITE
#file2 = open('file_assist.txt', "a")
#file2.write('This is a new file')
#file2.close()

#file3 = open('file_assist.txt', "w")
#file3.write('This is the write method')
#file3.close()

#file2 = open('file_assist.txt', "r")
#print(file2.read())
#file2.close()

#CREATE
file3 = open('.txt', 'x')


 
