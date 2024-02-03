#python exemption error handling is used to handle errors in python.
#the triblock lets you test your code for error: uses the "try" inbuilt function
#the except block lets us check the error. The except helps us to name the error using the "except" function
#the else block
#if the variable is defined, it prints but if not the except blocks shows why it was unable to be executed

#x = "how are you"
#try:
 #   print(x)
#except:
 #   print('X IS NOT DEFINED')

#there are different types of error like syntax error and indentation error. To make an exception gotten from a code, we stipulate it in te code
#since print(x) is a name kind, it is a NameError so its exception will take the print function under the NameError
try:
    print(x)
except SyntaxError:
    print('This is a syntax error')
except IndentationError:
    print('X is not defined')
except:
    print('something went wrong')