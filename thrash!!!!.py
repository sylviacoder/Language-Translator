print("Welcome to Goldmine Bank Dear Customer!!!\n\nPlease Insert your Card")

class ATM:
    def __init__(self, pin, attempts):
        self.pin = 1234
        self.attempts = 3

    def check_pin(self, entered_pin):
        if self.attempts >= 3:
            return "Account locked. Please contact the bank."
        if entered_pin == self.pin:
            return "Access granted."
        else:
            self.attempts += 1
            return f"Incorrect pin. {3 - self.attempts} attempts remaining."

def main():
    pin = input("Enter your pin: ")
    atm = ATM(pin)
    while True:
        entered_pin = input("Enter pin: ")
        result = atm.check_pin(entered_pin)
        print(result)
        if "Access granted" in result:
            break
        elif "Account locked" in result:
            break

if __name__ == "__main__":
    main()