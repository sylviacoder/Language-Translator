#ENCAPSULATION
class BankAccount:
    def __init__(self, balance=0):
        self._balance = balance

    def get_balance(self):
        return self._balance
    
    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self._balance -= amount
        else:
            print("Insufficient Funds")

my_account = BankAccount(1000)
my_account.withdraw(500)
my_account.deposit(5000)
print(f"Remaining balance: {getmy_account.get_balance()}")