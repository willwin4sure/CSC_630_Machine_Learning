import functools

@functools.lru_cache()
def collatz(n):
    if (n < 1 or int(n) != n):
        print("You did not enter a positive integer.")
    elif n == 1:
        return [1]
    else:
        if n % 2 == 0:
            return [n] + collatz(n//2)
        else:
            return [n] + collatz(3*n+1)

while(True):
    num = input("Enter the positive integer you want the Collatz sequence of: ")
    print(collatz(int(num)))

