print("Welcome to Goldmine Bank Dear Customer!!!\n\nPlease Insert your Card")
Attempts = 3 #numbers of PIN attempts
CustomerPin = 1234
Balance = 500000

while Attempts != 0:
    pincode = int(input("Please Enter your 4 Digits Pin "))
    if pincode != CustomerPin:
        Attempts -= 1
        print("Wrong Pin!\nYou have", Attempts, "Trials left")
    else:
        Customer_choice = input("**** MENU ****\n1 = Balance\n2 = Deposit\n3 = Withdrawal")
        if Customer_choice == "1":
            print("Your Total Balance is", Balance)

        if Customer_choice == "2":
            Customer_Deposit = int(input("Enter the Amount you Wish to Deposit: "))
            NewBalance = Customer_Deposit + Balance
            print(f"Dear Customer, You have Successfully Deposited ${Customer_Deposit} into your Account\n Your Balance is ${NewBalance}")

        if Customer_choice == "3":
            Customer_Withdraw = int(input("Enter the Amount you Would like to Withdraw: "))
            if Balance >= Customer_Withdraw:
                NewBalance = Balance - Customer_Withdraw
                print(f"Successful!!! You have withdrawn {Customer_Withdraw} from your account\n Your New Balance is {NewBalance}")
            else:
                print("Insufficient Funds\nDear Customer you do not have enough in your account")

        print("\nPress any Key to Continue")
        Exit = input("Would you like to perform another transaction? Y/N")
        if Exit.lower() == "y":
                 continue
#        elif Exit.upper() == "Y":
 #
 #             break
       
        else:
             print("Please take your card\n\nThank you for Banking with us!")
             
             
        

    

    






    