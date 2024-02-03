#capsulation: means restricting object components. It hides components in order to restrict access to it.
class Classname:
    def __init__(self, data):
        self.data = 'test'
    def displayActive():
        print("ACTIVE MR SEUN")
        
    
class Derived(Classname):
    def __init__(self):
        
        print(f'protected attribute of base classname: {self.data}')
    
        self.data = 'changed'
        print(f'modified attribute: {self.data}')
    
test2 = Classname()
print(test2.displayActive)
