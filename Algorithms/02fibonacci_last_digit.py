# Uses python3
# Compute the last digit of Fibnacci Number
import sys
def get_fibonacci_last_digit(n):
   if (n <= 1):
       return n
   F = [0]*(n+1)
   F[1] = 1
   # Only save the last digit of Fibnacci
   for i in range(2, n+1):
       F[i] = (F[i-1] + F[i-2]) % 10
   return F[n]

if __name__ == '__main__':
    input = sys.stdin.read()
    n = int(input)
    print(get_fibonacci_last_digit(n))
