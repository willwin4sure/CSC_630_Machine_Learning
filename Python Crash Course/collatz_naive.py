
def print_collatz(n):
    while (n != 1):
        print(n, end = ' ')
        if (n % 2 == 0):
            n = n // 2
        else:
            n = 3*n + 1
    print(1)

num = input("Enter the number whose Collatz sequence you want to print: ")
try:
    num = int(num)
    print_collatz(num)
except:
    print("Error. You likely didn't input a positive integer.")

