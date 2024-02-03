class CustomerCard():
    def __init__(self, CardNum, pincode, firstname, lastname, balance):
        self.CardNum = CardNum
        self.pincode = pincode
        self.firstname = firstname
        self.lastname = lastname
        self.balance = balance


    def get_CardNum(self):
        return self.CardNum
    def get_pincode(self):
        return self.pincode
    def get_firstname(self):
        return self.firstname
    def get_lastname(self):
        return self.lastname
    def get_balance(self):
        return self.balance
    
    def set_CardNum(self, reset):
        self.CardNum = reset
    def set_pincode(self, reset):
        self.pincode = reset
    def set_firstname(self, reset):
        self.firstname = reset
    def set_lastname(self, reset):
        self.lastname = reset
    def set_balance(self, reset):
        self.balance = reset

    def print_out(self):
        print("Card #: ", self.CardNum)
        print("Pincode: ", self.pincode)
        print("First Nmae: ", self.firstname)
        print("Last Name: ", self.lastname)
        print("Balance: ", self.balance)