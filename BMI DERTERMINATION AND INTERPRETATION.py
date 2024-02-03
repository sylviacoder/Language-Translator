#BMI DERTERMINATION AND INTERPRETATION
print('Welcome to Magic land, kindly fill your information and have a magical time!')
weight = float(input('enter your weight in kg: '))
height = float(input('enter your height in meter: '))

def your_bmi():
    bmi = weight / (height)**2
    return bmi
    
bmi = your_bmi()
print(f'Dear customer, your bmi is {bmi}')

if bmi < 18.5:
    print ('Error! You are underweight for this ride')
elif bmi >= 18.5 and bmi < 25:
    print ('Great! You met the normal weight for this ride')
elif bmi >= 25 and bmi < 30:
    print ('Oops! You are overweight for this ride')
elif bmi >= 30 and bmi < 35:
    print ('Sorry! You are obese and do not meet the appropriate weight for this ride')
else:
    print('Clinically obese')    
    
#LEAP YEAR
year = int(input('Enter a year: '))

if(year % 400 == 0) and (year % 100 == 0):
    print(f' The year {year} is a leap year')
elif(year % 4 == 0) and (year % 100 != 0):
    print(f' The year {year} is a leap year')
else:
    print(f'The year {year} is not a leap year')

#A program that calculates the highest score
student_score = [72, 56, 98, 67, 54, 72, 91, 80]

highest_score = 0
for score in student_score:
    if score > highest_score:
        highest_score = score
         
print('The highest score is', highest_score)

#A for loop that prints the even numbers from 0-500
for numbers in range (0, 501, 2):
    print(numbers)    
#employee input
def show_employee():
    fname = input('Enter your name? ')
    lname = input('Enter your last name? ')
    number = input('Enter your monthly salary in dollars? ')
    print(f'Finish! The owner of this employee card is {fname} {lname} and she is paid {number} monthly')
    
show_employee()
#Write a program to create a function calculation() such that it can accept two variables and calculate the addition and subtraction. user input is required
def calculation():
    first_digit = int(input('enter a number: '))
    second_digit = int(input('enter a second number: '))
    total = first_digit + second_digit
    deduction = second_digit - first_digit
    print(f'the sum output is: {total}')
    print(f'the subtraction output is: {deduction}')

calculation()

#write a function with a default parameter
def diagnosis(ailment = 'ulcer'):
    print(f'the doctor told me it was {ailment}')
    
diagnosis('migraine')
diagnosis()
diagnosis('appollo')

#write a keyword argument function
def gadgets(phone2, phone9, phone4, phone7, phone5):  
    print(f'I would like to sell my {phone9}')

gadgets(phone2 = 'redmi', phone7 = 'samsung', phone9 = 'iphone', phone4 = 'nokia', phone5 = 'infinix')
    



    

    
    
    

    
