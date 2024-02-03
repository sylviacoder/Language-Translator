class ATM:
    def __init__(self, balance=0, pin=0000):
        self.balance = balance
        self.pin = pin

    def check_balance(self):
        return self.balance

    def withdraw(self, amount):
        if amount > self.balance:
            return "Insufficient funds."
        else:
            self.balance -= amount
            return f"Withdrawal successful. Current balance: {self.balance}"

    def deposit(self, amount):
        self.balance += amount
        return f"Deposit successful. Current balance: {self.balance}"

    def change_pin(self, new_pin):
        self.pin = new_pin
        return "PIN changed successfully."

    def main(self):
        while True:
            print("Welcome to the ATM machine.")
            pin = int(input("Please enter your PIN: "))

            if pin != self.pin:
                print("Incorrect PIN. Please try again.")
                continue

            print("\nMenu:")
            print("1. Check Balance")
            print("2. Withdraw")
            print("3. Deposit")
            print("4. Change PIN")
            print("5. Exit")

            choice = int(input("Please enter your choice: "))

            if choice == 1:
                print(self.check_balance())
            elif choice == 2:
                amount = int(input("Please enter the amount to withdraw: "))
                print(self.withdraw(amount))
            elif choice == 3:
                amount = int(input("Please enter the amount to deposit: "))
                print(self.deposit(amount))
            elif choice == 4:
                new_pin = int(input("Please enter your new PIN: "))
                print(self.change_pin(new_pin))
            elif choice == 5:
                print("Exiting the ATM machine.")
                break
            else:
                print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()