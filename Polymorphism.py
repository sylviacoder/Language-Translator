#import re
#The word "polymorphism" means "many forms", and in programming it refers to methods/functions/operators with the same name that can be executed on many objects or classes
#An example of a Python function that can be used on different objects is the len() 

#String
#for string len() returns the number of characters
x = "Hello World!"

print(len(x))
      
#TUPLE
#For tuples len() returns the number of items in the tuple
mytuple = ("apple", "banana", "cherry")

print(len(mytuple))

#Dictionary
#For dictionaries len() returns the number of key/value pairs in the dictionary
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}

print(len(thisdict))

#class polymorphism  is often used in Class methods, where we can have multiple classes with the same method name.
#For example, say we have three classes: Car, Boat, and Plane, and they all have a method called move()
class car:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def move(self):
        print("Drive")

class boat:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def move(self):
        print("Sail")

class plane:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model

    def move(self):
        print("Fly")

car1 = car("Ford", "Mustang")       #Create a Car class
boat1 = boat("Ibiza", "Touring 20") #Create a Boat class
plane1 = plane("Boeing", "747")     #Create a Plane class

for x in (car1, boat1, plane1):
  x.move()

#Inheritance polymorphism
#make a parent class called Vehicle, and make Car, Boat, Plane child classes of Vehicle, the child classes inherits the Vehicle methods, but can override them
class Vehicle:
  def __init__(self, brand, model):
    self.brand = brand
    self.model = model

  def move(self):
    print("Move!")

class Car(Vehicle):
  pass

class Boat(Vehicle):
  def move(self):
    print("Sail!")

class Plane(Vehicle):
  def move(self):
    print("Fly!")

car1 = Car("Ford", "Mustang") #Create a Car object
boat1 = Boat("Ibiza", "Touring 20") #Create a Boat object
plane1 = Plane("Boeing", "747") #Create a Plane object

for x in (car1, boat1, plane1):
  print(x.brand)
  print(x.model)
  x.move()