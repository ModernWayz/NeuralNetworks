# Unfinished shit

from asyncore import write
stopped = False
while stopped == False:
    fileName = input("Bestandnaam? ")
    f = open(fileName, 'w')
    print("Actie?")
    print("(1) Schrijven")
    print("(2) Lezen")
    print("(3) Toevoegen")

    choice = int(input("Keuze: "))

    if choice == 1:
        writing = True
        fullText = ""
        while writing == True:
            fullText += input("Welke tekst moet er geschreven worden? ")
            stopWriting = input("Verdergaan? (J/N) ")
            if stopWriting == 'N':
                writing = False
        f.write(fullText)
    elif choice == 2:
        fileText = f.read()
        print(fileText)
    elif choice == 3:
        print("")
    else:
        break

